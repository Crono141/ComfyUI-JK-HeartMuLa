# ComfyUI-JK-HeartMuLa

**HeartMuLa music generation + lyrics transcription, with MuQ-MuLan reference-audio style transfer.**

This is a fork of [BobRandomNumber/ComfyUI-HeartMuLa](https://github.com/BobRandomNumber/ComfyUI-HeartMuLa) (Apache-2.0) that adds the ability to condition generation on the *style* of a reference track — genre, mood, instrumentation — via a MuQ-MuLan audio embedding. It bundles the full HeartMuLa toolset (loaders, generator, decoder, post-processor, transcription) so it works as a complete, standalone pack.

All node ids are uniquely prefixed (`JKHeartMuLa*` for reused nodes, `HML*` for the style-transfer additions) and grouped under the **`JK-HeartMuLa`** category, so **this pack can be installed alongside the original ComfyUI-HeartMuLa** with no conflicts.

## What style transfer does (and doesn't)

It captures **genre, mood, and instrumentation** from the reference audio. It does **not** clone voice timbre or melody — MuQ-MuLan produces a single global style embedding, not a time-aligned one.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Crono141/ComfyUI-JK-HeartMuLa.git
cd ComfyUI-JK-HeartMuLa
pip install -r requirements.txt
```

Restart ComfyUI and search the node menu under **JK-HeartMuLa**.

## Model Setup

Weights are **not** auto-downloaded (to avoid duplicate/scattered models). Place them manually, exactly as for the original HeartMuLa pack. Create a base folder `ComfyUI/models/HeartMuLa/` and organize:

### 1. Base configuration files (in the root of `HeartMuLa/`)
- [gen_config.json](https://huggingface.co/HeartMuLa/HeartMuLaGen/blob/main/gen_config.json)
- [tokenizer.json](https://huggingface.co/HeartMuLa/HeartMuLaGen/blob/main/tokenizer.json)

### 2. Model / codec directories (subfolders of `HeartMuLa/`)
| Use | Model | Codec |
|---|---|---|
| ⭐ Recommended | [HeartMuLa-oss-3B-happy-new-year](https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B-happy-new-year) | [HeartCodec-oss-20260123](https://huggingface.co/HeartMuLa/HeartCodec-oss-20260123) |
| Standard | [HeartMuLa-oss-3B](https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B) | [HeartCodec-oss](https://huggingface.co/HeartMuLa/HeartCodec-oss) |
| RL-tuned | [HeartMuLa-RL-oss-3B-20260123](https://huggingface.co/HeartMuLa/HeartMuLa-RL-oss-3B-20260123) | [HeartCodec-oss-20260123](https://huggingface.co/HeartMuLa/HeartCodec-oss-20260123) |
| Transcription | [HeartTranscriptor-oss](https://huggingface.co/HeartMuLa/HeartTranscriptor-oss) | — |

```text
ComfyUI/models/HeartMuLa/
├── gen_config.json
├── tokenizer.json
├── HeartMuLa-oss-3B-happy-new-year/
├── HeartCodec-oss-20260123/
└── ...
```

The MuQ-MuLan style model (`OpenMuQ/MuQ-MuLan-large`, ~2.5 GB) is the one exception — it downloads automatically from Hugging Face the first time the **HML MuQ Model Loader** node runs (it has no fixed ComfyUI models folder). Loader nodes include a 📁 folder browser for picking `base_path`.

## Nodes

**Music generation (reused from HeartMuLa, rebranded):**
- **HeartMuLa Tag Builder** — per-category fields (genre / mood / instrument / vocal / production free-text with tag-suggestion tooltips, tempo / era dropdowns) plus a **multiline** `additional_tags` box. Lowercases, de-dupes, and comma-joins into a single `tags` string for the Music Generator. (Original implementation; concept inspired by RT-HeartMuLa.)
- **HeartMuLa Loader** — loads the generator LLM. `base_path`, `model_version`, `torch_compile` + backend/mode.
- **HeartMuLa Codec Loader** — loads the audio codec (fp32).
- **HeartMuLa Music Generator** — the core generator. Same controls as HeartMuLa's (lyrics, tags, duration, seed, temperature, top_k, cfg_scale) **plus an optional `cmuq` input** for style transfer. Leave `cmuq` unconnected and it behaves exactly like the stock generator. Outputs tokens.
- **HeartMuLa Audio Decoder** — tokens → 48 kHz audio.
- **Audio Post-Processor** — normalize / stereo width / high-pass / low-pass / gain.

**Lyrics transcription (reused):**
- **HeartMuLa Transcription Loader** / **HeartMuLa Lyrics Transcriber** — Whisper-based audio → text.

**Style transfer (new in this fork):**
- **HeartMuLa MuQ Model Loader** — loads `OpenMuQ/MuQ-MuLan-large` on CPU (~2.5 GB RAM, no VRAM). Singleton; loads once per session.
- **HeartMuLa Style Embed** — produces the 512-D style embedding (24 kHz mono) from a reference clip, with a per-clip progress bar. Provide the reference either by **dragging an audio file onto the node** (or the upload button — the file lands in ComfyUI's `input/` folder) or by wiring the optional **`audio_input` socket** from any AUDIO source (Load Audio, Record Audio, a trimmed clip, generated audio…). The socket takes priority when connected. `style_strength` (0–10) scales influence: `0` = off (identical to no reference), `1.0` = natural, higher = stronger (and eventually unstable).

## Basic workflows

Style transfer (`example_workflows/style_transfer_basic.json`):
```
HeartMuLa MuQ Model Loader → HeartMuLa Style Embed (drop reference audio) ─┐
                                                                           ▼ (cmuq, optional)
HeartMuLa Loader ──────────────────────────────► HeartMuLa Music Generator → tokens ─┐
                                                                                      ▼
HeartMuLa Codec Loader ───────────────────────► HeartMuLa Audio Decoder → Save Audio (→ media assets)
```

Also included: `HeartMuLaGeneration.json` (plain generation) and `HeartMuLaTranscription.json` (lyrics transcription).

## Credits & license

This pack is Apache-2.0, as a derivative of:
- [BobRandomNumber/ComfyUI-HeartMuLa](https://github.com/BobRandomNumber/ComfyUI-HeartMuLa) — base nodes and bundled `heartlib`. See `NOTICE` for the list of modifications.
- [HeartMuLa/heartlib](https://github.com/HeartMuLa/heartlib) — the model library.
- [OpenMuQ/MuQ](https://github.com/tencent-ailab/MuQ) — the MuQ-MuLan style-embedding model.

```bibtex
@misc{yang2026heartmulafamilyopensourced,
      title={HeartMuLa: A Family of Open Sourced Music Foundation Models},
      author={Dongchao Yang and Yuxin Xie and Yuguo Yin and others},
      year={2026},
      eprint={2601.10547},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2601.10547},
}
```
