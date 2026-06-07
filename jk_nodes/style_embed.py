"""HeartMuLa Style Embed.

Extracts a 512-D MuQ-MuLan style embedding from reference audio and scales it by
``style_strength``. The result (the custom ``JKHEARTMULA_CMUQ`` type) feeds
HeartMuLa Music Generator, where it replaces the zeroed ``muq_embed`` conditioning
vector that heartlib uses for tag-only generation.

Two ways to provide the reference audio:
  - the **audio upload widget** (drag a file onto the node / use the picker; same
    mechanism as core Load Audio -- the file is stored in ComfyUI's ``input/``
    folder, so it shows up in the input media assets), or
  - the optional **``audio_input`` socket** (AUDIO), so the node drops into any
    workflow fed by Load Audio, Record Audio, a trimmed clip, generated audio, etc.

If the ``audio_input`` socket is connected it takes priority over the uploaded file.

Note: the upload widget must be named ``audio`` -- ComfyUI's core audio-upload
integration keys on that exact input name -- so the socket is ``audio_input``.

MuQ-MuLan expects 24 kHz mono audio. The generator's output sample rate is
unrelated (HeartCodec decodes at 48 kHz); 24 kHz here is only the rate the
embedding model was trained on.
"""

import hashlib
import os

import torch

import folder_paths
from comfy_api.latest import io

from .muq_loader import MUQ_TYPE, ensure_on_desired_device, offload_muq

# Custom socket type for the style embedding. Matches the optional input type on
# HeartMuLa Music Generator.
CMUQ_TYPE = io.Custom("JKHEARTMULA_CMUQ")

# MuQ-MuLan-large embedding dimension (== HeartMuLa's muq_dim). A zero vector of
# this size means "no style", identical to heartlib's default muq_embed.
MUQ_EMBED_DIM = 512


class JKHeartMuLaStyleEmbed(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        input_dir = folder_paths.get_input_directory()
        os.makedirs(input_dir, exist_ok=True)
        files = folder_paths.filter_files_content_types(os.listdir(input_dir), ["audio", "video"])
        return io.Schema(
            node_id="JKHeartMuLaStyleEmbed",
            display_name="HeartMuLa Style Embed",
            category="JK-HeartMuLa",
            inputs=[
                MUQ_TYPE.Input("muq_model"),
                io.Combo.Input(
                    "audio",
                    upload=io.UploadType.audio,
                    options=sorted(files),
                    tooltip="Reference audio (drag-drop or upload). Ignored if the "
                            "'audio_input' socket is connected.",
                ),
                io.Float.Input(
                    "style_strength",
                    default=1.0,
                    min=0.0,
                    max=10.0,
                    step=0.05,
                    display_mode=io.NumberDisplay.slider,
                ),
                # Appended after the original widgets (not inserted) so existing
                # saved workflows keep their widget positions and still load.
                io.Boolean.Input(
                    "enable",
                    default=True,
                    tooltip="When off, output a zero embedding (no style transfer) -- "
                            "identical to style_strength = 0. Lets a single switch toggle "
                            "style transfer on/off in a shared workflow without rewiring.",
                ),
                io.Boolean.Input(
                    "free_vram_after",
                    default=True,
                    tooltip="After embedding, offload MuQ-MuLan back to CPU and free its VRAM "
                            "so the HeartMuLa models have room to generate. No effect when MuQ "
                            "runs on CPU. It stays cached in RAM (no reload), and is moved back "
                            "to the chosen device automatically on the next embed.",
                ),
                # Optional: feed reference audio from another node (Load Audio,
                # Record Audio, a trimmed clip, generated audio...). Overrides the
                # uploaded file when connected.
                io.Audio.Input("audio_input", optional=True),
            ],
            outputs=[CMUQ_TYPE.Output(display_name="cmuq")],
        )

    @classmethod
    def fingerprint_inputs(cls, audio=None, style_strength=1.0, enable=True,
                           free_vram_after=True, **kwargs) -> str:
        # Re-run when any toggle, the strength, or the uploaded file content
        # changes. (Changes to a connected `audio_input` socket are detected via
        # the normal input graph.)
        base = f"{style_strength}:{enable}:{free_vram_after}"
        try:
            path = folder_paths.get_annotated_filepath(audio)
            m = hashlib.sha256()
            with open(path, "rb") as f:
                m.update(f.read())
            return f"{m.hexdigest()}:{base}"
        except Exception:
            return f"{audio}:{base}"

    @classmethod
    def execute(cls, muq_model, audio, style_strength, enable=True,
                free_vram_after=True, audio_input=None) -> io.NodeOutput:
        # When disabled, emit a zero embedding (no style transfer) and skip all
        # audio loading / MuQ work -- a single switch to toggle style in a shared
        # workflow. Equivalent to style_strength = 0.
        if not enable:
            print("[JK-HeartMuLa] Style Embed disabled -> zero embedding (no style transfer)")
            if free_vram_after:
                offload_muq()
            return io.NodeOutput(torch.zeros(MUQ_EMBED_DIM, dtype=torch.bfloat16))

        # Make sure MuQ is on its selected device (it may have been offloaded to
        # CPU after a previous run).
        muq_model = ensure_on_desired_device() or muq_model

        # Resolve the reference waveform to a 24 kHz mono 1-D tensor. A connected
        # audio_input socket takes priority over the uploaded file.
        if audio_input is not None:
            import torchaudio

            waveform = audio_input["waveform"]          # [batch, channels, samples]
            sample_rate = int(audio_input["sample_rate"])
            wav_t = waveform[0].mean(dim=0).float().cpu()  # mono, first batch item
            if sample_rate != 24000:
                wav_t = torchaudio.functional.resample(wav_t, sample_rate, 24000)
            wav_t = wav_t.contiguous()
            print(f"[JK-HeartMuLa] Extracting style embedding from connected audio "
                  f"input ({sample_rate} Hz)")
        else:
            import librosa

            if not audio:
                raise ValueError(
                    "No reference audio. Connect the 'audio_input' socket, or drag "
                    "an audio file onto the node / use the upload button.")
            audio_path = folder_paths.get_annotated_filepath(audio)
            print(f"[JK-HeartMuLa] Extracting style embedding from: {audio_path}")
            wav, _ = librosa.load(audio_path, sr=24000, mono=True)
            wav_t = torch.from_numpy(wav).float()

        with torch.no_grad():
            embedding = cls._embed_with_progress(muq_model, wav_t)

        # Return a small CPU bfloat16 vector regardless of where MuQ ran; the
        # generator moves it onto the generation device as needed.
        embedding = embedding.squeeze(0).to(device="cpu", dtype=torch.bfloat16)  # [512]
        embedding = embedding * style_strength

        print(f"[JK-HeartMuLa] Style embedding extracted, strength={style_strength}")

        # Free MuQ's VRAM for generation (no-op if it ran on CPU).
        if free_vram_after:
            offload_muq()

        return io.NodeOutput(embedding)

    @staticmethod
    def _embed_with_progress(muq_model, wav_t):
        """Return the MuQ-MuLan embedding, driving a per-clip ComfyUI progress bar.

        MuQ-MuLan splits audio >10s into clips and averages their latents. We
        replicate that loop (using MuQ's own clip splitter + latent function) so
        the node shows real progress -- the result is numerically identical to a
        one-shot ``muq_model(wavs=...)`` call. Falls back to the one-shot call if
        a future muq version changes these internals.
        """
        import comfy.utils

        # Run the encoder on whichever device the MuQ model is loaded on.
        try:
            dev = next(muq_model.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")

        try:
            clips = muq_model._get_all_clips(wav_t)  # [n_clips, clip_samples]
            n_clips = int(clips.shape[0])
            pbar = comfy.utils.ProgressBar(n_clips)
            latents = []
            for i in range(n_clips):
                clip = clips[i].unsqueeze(0).to(dev)  # [1, clip_samples]
                latents.append(muq_model.mulan_module.get_audio_latents(clip).squeeze(0))
                pbar.update(1)
            return torch.stack(latents, dim=0).mean(dim=0).unsqueeze(0)  # [1, 512]
        except Exception as e:
            print(f"[JK-HeartMuLa] Per-clip progress unavailable ({e}); "
                  f"falling back to one-shot embedding.")
            return muq_model(wavs=wav_t.unsqueeze(0).to(dev))  # [1, 512]
