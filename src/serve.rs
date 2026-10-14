//! HTTP server: `POST /v1/speak` renders text to WAV (whole, or streamed while it is generated).
//!
//! The GPU engine is single-tenant, so one worker thread owns the [`Tts`] and requests queue for it; the ntex
//! handlers only parse requests and forward audio. Endpoints:
//!
//! - `POST /v1/speak`: JSON `{"text": ..., "voice": "name", "stream": false, "temperature": 1.0, "top_k": 30,
//!   "top_p": 0.9, "seed": 0, "split": true, "max_new": 2048}` (only `text` is required), or the text itself as a
//!   plain body. For dialogue, `"voices": ["alice", "bob"]` clones alice as `<|speaker:0|>` and bob as
//!   `<|speaker:1|>` (or `{"2": "alice"}` to choose the numbers); speaker tags and inline tags like `[laughs]` in
//!   `text` work as on the command line. Returns `audio/wav`, 44.1 kHz mono 16-bit. With `"stream": true` (or `?stream=1`) the WAV is sent
//!   as it is generated, with an open-ended header that players such as `pw-play`, `ffplay` and `mpv` accept.
//! - `POST /v1/voices?name=NAME&text=TRANSCRIPT`: body is a WAV clip; registers a voice to clone.
//! - `GET /v1/voices`: registered voice names. `GET /health`: `ok`.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::mpsc as std_mpsc;
use std::task::{Context, Poll};

use anyhow::{anyhow, Result};
use ntex::http::StatusCode;
use ntex::util::{Bytes, Stream};
use ntex::web::{self, App, HttpRequest, HttpResponse, HttpServer};
use serde::Deserialize;
use tokio::sync::{mpsc, oneshot};

use crate::codec::SAMPLE_RATE;
use crate::engine::SampleParams;
use crate::prompt::Reference;
use crate::tts::{SpeakOpts, Tts};

/// Generation defaults for requests that leave a field out.
#[derive(Clone, Copy)]
pub struct Defaults {
    pub params: SampleParams,
    pub max_new: usize,
    pub first_chunk: usize,
    pub chunk: usize,
    pub split: bool,
}

#[derive(Deserialize, Default)]
struct SpeakRequest {
    text: String,
    voice: Option<String>,
    /// Several voices for multi-speaker text: `["alice", "bob"]` makes alice speaker 0 and bob speaker 1, or
    /// `{"0": "alice", "3": "bob"}` picks the speaker numbers
    voices: Option<serde_json::Value>,
    #[serde(default)]
    stream: bool,
    temperature: Option<f32>,
    top_k: Option<i32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    split: Option<bool>,
    max_new: Option<usize>,
}

#[derive(Deserialize)]
struct SpeakQuery {
    stream: Option<String>,
    voice: Option<String>,
}

#[derive(Deserialize)]
struct VoiceQuery {
    name: String,
    text: String,
}

type Chunk = Result<Bytes, std::io::Error>;

enum Job {
    Speak { req: SpeakRequest, out: mpsc::UnboundedSender<Chunk>, started: oneshot::Sender<Result<()>> },
    AddVoice { name: String, text: String, wav: Bytes, done: oneshot::Sender<Result<()>> },
    Voices { done: oneshot::Sender<Vec<String>> },
}

fn to_pcm16(audio: &[f32]) -> Bytes {
    let mut b = Vec::with_capacity(audio.len() * 2);
    for &s in audio {
        b.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
    }
    Bytes::from(b)
}

/// A 44-byte WAV header for 16-bit mono at [`SAMPLE_RATE`]; `data_len` `None` = unknown (streaming).
fn wav_header(data_len: Option<u32>) -> Bytes {
    let data = data_len.unwrap_or(u32::MAX - 36);
    let mut h = Vec::with_capacity(44);
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&(36u32.wrapping_add(data)).to_le_bytes());
    h.extend_from_slice(b"WAVEfmt ");
    h.extend_from_slice(&16u32.to_le_bytes());
    h.extend_from_slice(&1u16.to_le_bytes()); // PCM
    h.extend_from_slice(&1u16.to_le_bytes()); // mono
    h.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    h.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes());
    h.extend_from_slice(&2u16.to_le_bytes());
    h.extend_from_slice(&16u16.to_le_bytes());
    h.extend_from_slice(b"data");
    h.extend_from_slice(&data.to_le_bytes());
    Bytes::from(h)
}

/// `transcript` with its leading speaker tag set to `speaker`, so the reference clones that speaker.
fn retag(transcript: &str, speaker: u32) -> String {
    let t = transcript.trim_start();
    let rest = match crate::prompt::leading_speaker(t) {
        Some(_) => t.find("|>").map(|i| &t[i + 2..]).unwrap_or(t).trim_start(),
        None => t,
    };
    format!("<|speaker:{speaker}|> {rest}")
}

/// The references a request asks for: `voice` (as registered), and/or `voices` (list or speaker map).
fn resolve_voices(req: &SpeakRequest, voices: &HashMap<String, Reference>) -> Result<Vec<Reference>> {
    let get = |name: &str| voices.get(name).ok_or_else(|| anyhow!("unknown voice {name:?} (POST /v1/voices to add it)"));
    let mut refs = Vec::new();
    if let Some(v) = &req.voice {
        refs.push(get(v)?.clone());
    }
    let pairs: Vec<(u32, String)> = match &req.voices {
        None => vec![],
        Some(serde_json::Value::Array(a)) => a
            .iter()
            .enumerate()
            .map(|(i, v)| v.as_str().map(|n| (i as u32, n.to_string())).ok_or_else(|| anyhow!("voices: names must be strings")))
            .collect::<Result<_>>()?,
        Some(serde_json::Value::Object(m)) => m
            .iter()
            .map(|(k, v)| {
                let n: u32 = k.parse().map_err(|_| anyhow!("voices: keys are speaker numbers, got {k:?}"))?;
                let name = v.as_str().ok_or_else(|| anyhow!("voices: names must be strings"))?;
                Ok((n, name.to_string()))
            })
            .collect::<Result<_>>()?,
        Some(_) => anyhow::bail!("voices: a list of names or {{\"speaker\": name}} map"),
    };
    for (speaker, name) in pairs {
        if speaker > 15 {
            anyhow::bail!("voices: speaker {speaker} out of range 0..15");
        }
        let mut r = get(&name)?.clone();
        r.transcript = retag(&r.transcript, speaker);
        refs.push(r);
    }
    Ok(refs)
}

/// The engine thread: runs jobs one at a time.
fn worker(mut tts: Tts, mut voices: HashMap<String, Reference>, defaults: Defaults, jobs: std_mpsc::Receiver<Job>) {
    for job in jobs {
        match job {
            Job::Voices { done } => {
                let mut names: Vec<String> = voices.keys().cloned().collect();
                names.sort();
                let _ = done.send(names);
            }
            Job::AddVoice { name, text, wav, done } => {
                let r = crate::codec::read_wav_bytes(&wav)
                    .and_then(|(audio, sr)| tts.reference_audio(&audio, sr, &text))
                    .map(|reference| {
                        eprintln!("voice {name:?} registered");
                        voices.insert(name, reference);
                    });
                let _ = done.send(r);
            }
            Job::Speak { req, out, started } => {
                let refs = match resolve_voices(&req, &voices) {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = started.send(Err(e));
                        continue;
                    }
                };
                let d = defaults;
                let opts = SpeakOpts {
                    params: SampleParams {
                        temperature: req.temperature.unwrap_or(d.params.temperature),
                        top_k: req.top_k.unwrap_or(d.params.top_k),
                        top_p: req.top_p.unwrap_or(d.params.top_p),
                        seed: req.seed.unwrap_or(d.params.seed),
                        ..d.params
                    },
                    max_new: req.max_new.unwrap_or(d.max_new),
                    first_chunk: d.first_chunk,
                    chunk: d.chunk,
                    split: req.split.unwrap_or(d.split),
                };
                let _ = started.send(Ok(()));
                let sink = |audio: &[f32]| -> Result<()> {
                    out.send(Ok(to_pcm16(audio))).map_err(|_| anyhow!("client went away"))
                };
                match tts.speak(&req.text, &refs, &opts, sink) {
                    Ok(s) => eprintln!(
                        "spoke {:.2}s of audio in {:.2}s ({} chars)",
                        s.audio.len() as f64 / SAMPLE_RATE as f64,
                        s.elapsed.as_secs_f64(),
                        req.text.chars().count()
                    ),
                    Err(e) => {
                        eprintln!("speak failed: {e:#}");
                        let _ = out.send(Err(std::io::Error::other(e.to_string())));
                    }
                }
            }
        }
    }
}

/// Body stream over the worker's audio chunks, optionally preceded by the WAV header.
struct AudioStream {
    header: Option<Bytes>,
    rx: mpsc::UnboundedReceiver<Chunk>,
}

impl Stream for AudioStream {
    type Item = Chunk;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Chunk>> {
        if let Some(h) = self.header.take() {
            return Poll::Ready(Some(Ok(h)));
        }
        self.rx.poll_recv(cx)
    }
}

fn error(status: StatusCode, msg: impl std::fmt::Display) -> HttpResponse {
    HttpResponse::builder(status).content_type("text/plain; charset=utf-8").body(format!("{msg}\n"))
}

async fn speak(jobs: std_mpsc::Sender<Job>, req: HttpRequest, body: Bytes) -> HttpResponse {
    let query = web::types::Query::<SpeakQuery>::from_query(req.query_string()).ok();
    let is_json = req.headers().get("content-type").and_then(|v| v.to_str().ok()).is_some_and(|v| v.contains("json"))
        || body.first() == Some(&b'{');
    let mut sreq = if is_json {
        match serde_json::from_slice::<SpeakRequest>(&body) {
            Ok(r) => r,
            Err(e) => return error(StatusCode::BAD_REQUEST, format!("bad JSON: {e}")),
        }
    } else {
        SpeakRequest { text: String::from_utf8_lossy(&body).into_owned(), ..Default::default() }
    };
    if let Some(q) = query {
        let q = q.into_inner();
        if let Some(s) = q.stream {
            sreq.stream = !matches!(s.as_str(), "0" | "false" | "no");
        }
        if q.voice.is_some() {
            sreq.voice = q.voice;
        }
    }
    if sreq.text.trim().is_empty() {
        return error(StatusCode::BAD_REQUEST, "no text");
    }
    let stream = sreq.stream;
    let (out, mut rx) = mpsc::unbounded_channel();
    let (started, started_rx) = oneshot::channel();
    if jobs.send(Job::Speak { req: sreq, out, started }).is_err() {
        return error(StatusCode::SERVICE_UNAVAILABLE, "engine stopped");
    }
    match started_rx.await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return error(StatusCode::BAD_REQUEST, e),
        Err(_) => return error(StatusCode::SERVICE_UNAVAILABLE, "engine stopped"),
    }
    if stream {
        return HttpResponse::Ok()
            .content_type("audio/wav")
            .streaming(AudioStream { header: Some(wav_header(None)), rx });
    }
    let mut pcm = Vec::new();
    while let Some(chunk) = rx.recv().await {
        match chunk {
            Ok(b) => pcm.extend_from_slice(&b),
            Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e),
        }
    }
    let mut wav = wav_header(Some(pcm.len() as u32)).to_vec();
    wav.extend_from_slice(&pcm);
    HttpResponse::Ok().content_type("audio/wav").body(wav)
}

async fn add_voice(jobs: std_mpsc::Sender<Job>, req: HttpRequest, body: Bytes) -> HttpResponse {
    let q = match web::types::Query::<VoiceQuery>::from_query(req.query_string()) {
        Ok(q) => q.into_inner(),
        Err(_) => return error(StatusCode::BAD_REQUEST, "need ?name=NAME&text=TRANSCRIPT and a WAV body"),
    };
    let (done, rx) = oneshot::channel();
    if jobs.send(Job::AddVoice { name: q.name, text: q.text, wav: body, done }).is_err() {
        return error(StatusCode::SERVICE_UNAVAILABLE, "engine stopped");
    }
    match rx.await {
        Ok(Ok(())) => HttpResponse::Ok().body("ok\n"),
        Ok(Err(e)) => error(StatusCode::BAD_REQUEST, format!("{e:#}")),
        Err(_) => error(StatusCode::SERVICE_UNAVAILABLE, "engine stopped"),
    }
}

async fn list_voices(jobs: std_mpsc::Sender<Job>) -> HttpResponse {
    let (done, rx) = oneshot::channel();
    if jobs.send(Job::Voices { done }).is_err() {
        return error(StatusCode::SERVICE_UNAVAILABLE, "engine stopped");
    }
    match rx.await {
        Ok(names) => HttpResponse::Ok().json(&names),
        Err(_) => error(StatusCode::SERVICE_UNAVAILABLE, "engine stopped"),
    }
}

/// Load the engine with `load` on a dedicated thread (CUDA state stays on the thread that created it), then serve
/// it on `addr` until the process is stopped (Ctrl-C).
pub fn run<F>(load: F, defaults: Defaults, addr: &str) -> Result<()>
where
    F: FnOnce() -> Result<(Tts, HashMap<String, Reference>)> + Send + 'static,
{
    let (jobs, rx) = std_mpsc::channel::<Job>();
    let (ready_tx, ready) = std_mpsc::channel::<Result<()>>();
    std::thread::Builder::new().name("drama-engine".into()).spawn(move || match load() {
        Ok((tts, voices)) => {
            let _ = ready_tx.send(Ok(()));
            worker(tts, voices, defaults, rx)
        }
        Err(e) => {
            let _ = ready_tx.send(Err(e));
        }
    })?;
    ready.recv().map_err(|_| anyhow!("engine thread died while loading"))??;
    serve(jobs, addr.to_string())?;
    Ok(())
}

#[ntex::main(name = "drama", signals = true)]
async fn serve(jobs: std_mpsc::Sender<Job>, addr: String) -> std::io::Result<()> {
    eprintln!("listening on http://{addr}  (POST /v1/speak, POST|GET /v1/voices, GET /health)");
    HttpServer::new(async move |_| {
        let (j1, j2, j3) = (jobs.clone(), jobs.clone(), jobs.clone());
        // voice clips are uploaded whole: allow a few minutes of 48 kHz stereo 16-bit WAV
        let cfg = web::WebAppConfig::new().set_state(web::types::PayloadConfig::new(64 << 20));
        App::new()
            .with_config(cfg)
            .service(web::resource("/health").to(async || "ok\n"))
            .service(web::resource("/v1/speak").route(web::post().to(move |req: HttpRequest, body: Bytes| speak(j1.clone(), req, body))))
            .service(
                web::resource("/v1/voices")
                    .route(web::post().to(move |req: HttpRequest, body: Bytes| add_voice(j2.clone(), req, body)))
                    .route(web::get().to(move || list_voices(j3.clone()))),
            )
    })
    .bind(addr.as_str(), ntex::SharedCfg::new("drama"))?
    .workers(1)
    .run()
    .await
}
