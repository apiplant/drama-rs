import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, IC, Section, FlagTable } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsCli() {
  return (
    <DocsLayout>
      <H1>drama</H1>
      <Lead>
        Speak text to a WAV file or the speakers, optionally in a voice cloned from a reference clip.
      </Lead>

      <Section>
        <H2>Examples</H2>
        <CopyBlock
          command={`drama speak --model drama-q4 --text "Hello there." --out hello.wav
drama speak --model drama-q4 --text "Hello there." --play      # stream to the speakers (pw-play)

# clone a voice from a reference clip and its transcript
drama speak --model drama-q4 --play \\
  --ref-wav voice.wav --ref-text "Exactly what is said in voice.wav." \\
  --text "<|speaker:0|> [sighs] Fine. If we miss it, it's on you."

drama speak --model drama-q4 --interactive --play             # keep the model loaded, speak each stdin line`}
        />
        <P>
          Multi-speaker text uses <IC>{"<|speaker:N|>"}</IC> tags (N in 0..15); inline tags like <IC>[laughs]</IC> are
          passed through.
        </P>
      </Section>

      <Section>
        <H2>speak</H2>
        <FlagTable
          rows={[
            { flag: "--model DIR", meaning: "The converted weights (apiplant/drama-q4, or tools/convert.py output)." },
            { flag: "--text TEXT", meaning: "Text to speak. With --interactive, lines are read from stdin instead." },
            { flag: "--out FILE.wav", meaning: "Write a 44.1 kHz WAV file." },
            { flag: "--play", meaning: "Stream the audio to the speakers (PipeWire) while it is generated." },
            { flag: "--ref-wav, --ref-text", meaning: "Reference clip(s) to clone, each with its transcript." },
            { flag: "--temperature 1.0, --top-k 30, --top-p 0.9", meaning: "Sampling, as the reference server; plus repetition-aware sampling over the last 16 frames." },
            { flag: "--seed 0", meaning: "Sampling seed." },
            { flag: "--max-len 4096, --chunk 512", meaning: "Context length (prompt rows + generated frames) and prefill chunk size." },
            { flag: "--first-chunk 4, --stream-chunk 16", meaning: "Frames in the first streamed audio chunk, and in each later one." },
            { flag: "--no-split", meaning: "Generate the whole text at once instead of sentence by sentence." },
          ]}
        />
        <P>
          Text is generated one sentence at a time, like the reference server's streaming endpoint. Without a voice
          reference, Drama 3 often never ends after a full stop when more text follows; speaking each sentence on its
          own avoids that, and a sentence that still stalls is cut at a length cap and said again.
        </P>
      </Section>

      <Section>
        <H2>Other commands</H2>
        <FlagTable
          rows={[
            { flag: "encode / decode", meaning: "The codec alone: WAV to codes and back." },
            { flag: "bench", meaning: "Measure decode speed." },
            { flag: "trace", meaning: "Run raw prompt rows and dump frames and logits per step (engine tests)." },
          ]}
        />
      </Section>
    </DocsLayout>
  );
}
