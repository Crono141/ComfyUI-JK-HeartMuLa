"""HeartMuLa Style Embed.

Extracts a 512-D MuQ-MuLan style embedding from a reference audio file and
scales it by ``style_strength``. The result (the custom ``JKHEARTMULA_CMUQ``
type) feeds HeartMuLa Music Generator, where it replaces the zeroed ``muq_embed``
conditioning vector that heartlib uses for tag-only generation.

The reference audio is chosen via an upload widget (drag a file onto the node,
or use the picker) -- the same mechanism as ComfyUI's core Load Audio node. The
file is uploaded into ComfyUI's ``input/`` folder, so it also shows up in the
input media assets.

MuQ-MuLan expects 24 kHz mono audio. The generator's output sample rate is
unrelated (HeartCodec decodes at 48 kHz); 24 kHz here is only the rate the
embedding model was trained on.
"""

import hashlib
import os

import torch

import folder_paths
from comfy_api.latest import io

from .muq_loader import MUQ_TYPE

# Custom socket type for the style embedding. Matches the optional input type on
# HeartMuLa Music Generator.
CMUQ_TYPE = io.Custom("JKHEARTMULA_CMUQ")


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
                io.Combo.Input("audio", upload=io.UploadType.audio, options=sorted(files)),
                io.Float.Input(
                    "style_strength",
                    default=1.0,
                    min=0.0,
                    max=10.0,
                    step=0.05,
                    display_mode=io.NumberDisplay.slider,
                ),
            ],
            outputs=[CMUQ_TYPE.Output(display_name="cmuq")],
        )

    @classmethod
    def fingerprint_inputs(cls, audio=None, style_strength=1.0, **kwargs) -> str:
        # Re-run when the reference file content or the strength changes. Hash the
        # file so re-uploading a different file under the same name is detected.
        try:
            path = folder_paths.get_annotated_filepath(audio)
            m = hashlib.sha256()
            with open(path, "rb") as f:
                m.update(f.read())
            return f"{m.hexdigest()}:{style_strength}"
        except Exception:
            return f"{audio}:{style_strength}"

    @classmethod
    def execute(cls, muq_model, audio, style_strength) -> io.NodeOutput:
        import librosa

        if not audio:
            raise ValueError(
                "No reference audio selected. Drag an audio file onto the node, "
                "or pick one with the upload button.")

        audio_path = folder_paths.get_annotated_filepath(audio)
        print(f"[JK-HeartMuLa] Extracting style embedding from: {audio_path}")

        # MuQ-MuLan requires 24 kHz mono input.
        wav, _ = librosa.load(audio_path, sr=24000, mono=True)
        wavs = torch.from_numpy(wav).unsqueeze(0).float()  # [1, samples]

        with torch.no_grad():
            embedding = muq_model(wavs=wavs)  # [1, 512]

        embedding = embedding.squeeze(0).to(torch.bfloat16)  # [512]
        embedding = embedding * style_strength

        print(f"[JK-HeartMuLa] Style embedding extracted, strength={style_strength}")
        return io.NodeOutput(embedding)
