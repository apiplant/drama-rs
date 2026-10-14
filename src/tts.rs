//! Text -> audio: prompt building, LM generation and streaming codec decode, overlapped.
//!
//! The LM runs on the calling thread; sampled frames go over a channel to a codec thread
//! (its own CUDA stream) that decodes them in small chunks and hands the audio to a sink
//! (speakers and/or a buffer) while generation continues.

use std::io::Write;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use tokenizers::Tokenizer;

use crate::codec::{self, Codec};
use crate::engine::{Engine, SampleParams};
use crate::model::Model;
use crate::prompt::{self, Reference};

pub struct Tts {
    pub engine: Engine,
    pub codec: Codec,
    tok: Tokenizer,
}

pub struct SpeakOpts {
    pub params: SampleParams,
    pub max_new: usize,
    /// Frames in the first streamed chunk (small = low latency) and in later ones.
    pub first_chunk: usize,
    pub chunk: usize,
}

pub struct Spoken {
    pub audio: Vec<f32>,
    pub frames: Vec<Vec<i32>>,
    pub prompt_rows: usize,
    /// The prompt rows [prompt_rows][1 + num_codebooks].
    pub prompt: Vec<i32>,
    pub first_audio: Option<Duration>,
    pub elapsed: Duration,
}

/// Raw f32 audio to the default output through PipeWire.
pub struct Player {
    child: Child,
}

impl Player {
    pub fn new() -> Result<Self> {
        let child = Command::new("pw-play")
            .args(["--raw", "--rate", &codec::SAMPLE_RATE.to_string(), "--channels", "1", "--format", "f32", "-"])
            .stdin(Stdio::piped())
            .spawn()
            .context("starting pw-play (PipeWire) for --play")?;
        Ok(Self { child })
    }

    pub fn write(&mut self, audio: &[f32]) -> Result<()> {
        let bytes = unsafe { std::slice::from_raw_parts(audio.as_ptr() as *const u8, std::mem::size_of_val(audio)) };
        self.child.stdin.as_mut().unwrap().write_all(bytes)?;
        Ok(())
    }

    /// Close the stream and wait until everything has been played.
    pub fn finish(mut self) -> Result<()> {
        drop(self.child.stdin.take());
        self.child.wait()?;
        Ok(())
    }
}

impl Tts {
    pub fn load(model_dir: &Path, max_len: usize, chunk: usize) -> Result<Self> {
        let codec = Codec::load(&model_dir.join("codec.safetensors"))?;
        let st = crate::gpu::Stream::new()?;
        let model = Model::load(model_dir, &st)?;
        let engine = Engine::new(model, max_len, chunk)?;
        let tok = Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer: {e}"))?;
        Ok(Self { engine, codec, tok })
    }

    /// Encode a reference clip (any rate, mixed to mono) for voice cloning.
    pub fn reference(&self, wav: &Path, transcript: &str) -> Result<Reference> {
        let (audio, sr) = codec::read_wav(wav)?;
        let audio = codec::resample(&audio, sr, codec::SAMPLE_RATE);
        let flat = self.codec.encode(&audio)?;
        let frames = flat.len() / 10;
        let codes = (0..10).map(|c| (0..frames).map(|t| flat[t * 10 + c]).collect()).collect();
        Ok(Reference { transcript: transcript.to_string(), codes })
    }

    /// Generate speech for `text`; `sink` receives audio chunks as soon as they are decoded.
    pub fn speak(&mut self, text: &str, refs: &[Reference], opts: &SpeakOpts, mut sink: impl FnMut(&[f32]) -> Result<()> + Send) -> Result<Spoken> {
        let dc = &self.engine.model.cfg.audio_decoder_config;
        let rows = prompt::build_prompt_rows(text, refs, &self.tok, dc.num_codebooks, dc.vocab_size)?;
        let width = dc.num_codebooks + 1;
        let prompt_rows = rows.len() / width;
        let t0 = Instant::now();
        let (tx, rx) = mpsc::channel::<Vec<i32>>();
        let codec = &self.codec;
        let engine = &mut self.engine;
        let (first, chunk) = (opts.first_chunk.max(1), opts.chunk.max(1));

        let (frames, decoded) = std::thread::scope(|scope| -> Result<_> {
            let worker = scope.spawn(move || -> Result<(Vec<f32>, Option<Duration>)> {
                let mut ds = codec.stream()?;
                let mut pending: Vec<i32> = Vec::new();
                let mut audio = Vec::new();
                let mut first_audio = None;
                let mut want = first;
                let mut flush = |pending: &mut Vec<i32>, audio: &mut Vec<f32>, first_audio: &mut Option<Duration>| -> Result<()> {
                    if pending.is_empty() {
                        return Ok(());
                    }
                    let a = codec.decode_chunk(&mut ds, pending)?;
                    pending.clear();
                    first_audio.get_or_insert_with(|| t0.elapsed());
                    sink(&a)?;
                    audio.extend_from_slice(&a);
                    Ok(())
                };
                while let Ok(codes) = rx.recv() {
                    pending.extend_from_slice(&codes);
                    // Take whatever else is already queued before decoding.
                    while let Ok(more) = rx.try_recv() {
                        pending.extend_from_slice(&more);
                    }
                    if pending.len() / 10 >= want {
                        flush(&mut pending, &mut audio, &mut first_audio)?;
                        want = chunk;
                    }
                }
                flush(&mut pending, &mut audio, &mut first_audio)?;
                Ok((audio, first_audio))
            });
            let frames = engine.generate(&rows, opts.params, opts.max_new, |frame| {
                let _ = tx.send(frame[1..].to_vec());
            });
            drop(tx);
            let decoded = worker.join().map_err(|_| anyhow!("codec thread panicked"))?;
            Ok((frames?, decoded?))
        })?;
        let (audio, first_audio) = decoded;
        Ok(Spoken { audio, frames, prompt_rows, prompt: rows, first_audio, elapsed: t0.elapsed() })
    }
}
