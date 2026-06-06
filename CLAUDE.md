# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this pack is

A **fork of `BobRandomNumber/ComfyUI-HeartMuLa`** (Apache-2.0) that adds MuQ-MuLan reference-audio style transfer to HeartMuLa music generation. It bundles the full HeartMuLa toolset so it is a complete standalone pack, and all node ids are uniquely prefixed so it **can be installed alongside the original** ComfyUI-HeartMuLa without collisions.

Distribution intent: a true GitHub fork at `Crono141/ComfyUI-JK-HeartMuLa` (preserves upstream history + Apache attribution). The folder still lives inside the parent ComfyUI git tree during development — it must become its own git root before publishing; do not commit it into the parent ComfyUI repo.

## Architecture

ComfyUI **V3 node API** throughout (`comfy_api.latest.io`: `define_schema`/`execute`, registered via a `ComfyExtension` + `comfy_entrypoint` in `__init__.py`). Not the legacy `NODE_CLASS_MAPPINGS` style.

- **`nodes.py`** — Bob's nodes, reused. Edited only to rebrand: every `node_id` is prefixed `JKHeartMuLa*` and `category` is `JK-HeartMuLa`. His `HeartMuLaMusicGenerator` class is removed (replaced by ours). Python class names are unchanged (e.g. `HeartMuLaLoader`), so `__init__.py` imports them by their original names.
- **`jk_nodes/`** — this fork's additions (node ids unified to the `JKHeartMuLa*` prefix, display names to `HeartMuLa *`, socket types `JKHEARTMULA_MUQ`/`JKHEARTMULA_CMUQ`):
  - `muq_loader.py` — `JKHeartMuLaMuQModelLoader`: lazy CPU singleton of `OpenMuQ/MuQ-MuLan-large` (~2.5 GB RAM). `fingerprint_inputs` returns a constant (V3 equivalent of `IS_CHANGED`).
  - `style_embed.py` — `JKHeartMuLaStyleEmbed`: reference audio via an **upload Combo widget** (`io.UploadType.audio`, drag-and-drop, same as core LoadAudio; file lands in `input/`) → 24 kHz mono (librosa) → 512-D embedding, scaled by `style_strength`. `fingerprint_inputs` hashes the file + strength.
  - `style_generator.py` — `JKHeartMuLaMusicGenerator`, **display name "HeartMuLa Music Generator"**. Replaces Bob's generator; a strict superset with an optional `cmuq` input.
- **`__init__.py`** — forked from Bob's: keeps the folder-browser server route (renamed to `/jkheartmula/browse`) and `WEB_DIRECTORY = "js"`; `get_node_list` returns the 6 rebranded nodes + our 3.
- **`js/heartmula_loader.js`** — folder browser. Targets the `JKHeartMuLa*` loader ids, extension renamed `JKHeartMuLa.FolderPicker`, fetches `/jkheartmula/browse`.
- **`heartlib/`** — bundled model library (from HeartMuLa/heartlib), unchanged.

### How style transfer works (resolved from heartlib source — do not guess)

heartlib has **no `cmuq`/`muq` generate kwarg**. Conditioning is a 512-D `muq_embed` vector that `HeartMuLaModel.preprocess()` **hardcodes to zeros** for tag-only generation, injected at `muq_idx` (right after the tags) and projected via `muq_linear` into the hidden state by `generate_tokens` → `generate_frame(continuous_segments=, starts=)`. Our generator calls `preprocess()` then **overwrites `processed_inputs["muq_embed"]`** with the reference embedding (matched to the existing tensor's batch/dtype). CFG is handled by the model (it overwrites the unconditional half itself).

Other resolved facts: generate→decode is **two nodes** (generator outputs `HEARTMULA_TOKENS`, decoder → `AUDIO` at **48 kHz**); the sampling kwarg is `topk` (not `top_k`); socket type strings (`HEARTMULA_MODEL/CODEC/TOKENS`, `AUDIO`) are **shared** with Bob's pack (intentional — only node ids must be unique).

## Build / verify

No build/lint/test tooling — a runtime ComfyUI plugin. Use the project venv (`/home/master/ComfyUI/venv/bin/python`); the system python has no torch.

```bash
pip install -r requirements.txt   # heartlib deps (torchao/torchtune/accelerate/vector-quantize-pytorch) + muq + librosa
```

Headless registration check (mirrors what ComfyUI's loader does — builds every node's schema):

```bash
cd /home/master/ComfyUI && ./venv/bin/python -c "import importlib.util,sys,os,asyncio; \
pkg='custom_nodes/ComfyUI-JK-HeartMuLa'; \
s=importlib.util.spec_from_file_location('jk',os.path.join(pkg,'__init__.py'),submodule_search_locations=[pkg]); \
m=importlib.util.module_from_spec(s); sys.modules['jk']=m; s.loader.exec_module(m); \
e=asyncio.get_event_loop().run_until_complete(m.comfy_entrypoint()); \
ns=asyncio.get_event_loop().run_until_complete(e.get_node_list()); \
[print(n.GET_SCHEMA().node_id) for n in ns]"
```

Then restart ComfyUI, hard-refresh the browser (node list is cached client-side), and look under the **JK-HeartMuLa** category (9 nodes).

## Conventions

- Model weights are **not** auto-downloaded (manual placement in `ComfyUI/models/HeartMuLa/`), except the MuQ-MuLan style model which pulls from HF on first loader run.
- Log lines from our additions are prefixed `[JK-HeartMuLa]`. Style embeddings are `bfloat16`; the MuQ model runs `float()` on CPU.
- Apache-2.0: keep `LICENSE` and `NOTICE`; record new modifications in `NOTICE`.
