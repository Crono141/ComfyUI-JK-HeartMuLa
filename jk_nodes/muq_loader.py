"""HeartMuLa MuQ Model Loader.

Lazy-loads OpenMuQ/MuQ-MuLan-large as a module-level singleton so it is only
loaded once per ComfyUI session. A `device` switch chooses where it runs:
  - "cpu"  -> ~2.5 GB system RAM, no VRAM (default).
  - "gpu"  -> faster style embedding, but ~2.5 GB VRAM stays resident (it is not
              freed between runs and competes with generation).
Style Embed runs the embedding on whichever device the model is loaded on.
"""

import threading

import comfy.model_management
import torch
from comfy_api.latest import io

# Custom socket type for the loaded MuQ-MuLan model. Matches the input type on
# HeartMuLa Style Embed.
MUQ_TYPE = io.Custom("JKHEARTMULA_MUQ")

_muq_model = None
_muq_loaded_choice = None  # "cpu" / "gpu" the singleton currently sits on
_muq_lock = threading.Lock()


def _resolve_device(device_choice):
    if device_choice == "gpu":
        return comfy.model_management.get_torch_device()
    return torch.device("cpu")


def _get_muq_model(device_choice):
    """Return the process-wide MuQ-MuLan singleton on the requested device."""
    global _muq_model, _muq_loaded_choice
    with _muq_lock:
        if _muq_model is None:
            from muq import MuQMuLan

            print("[JK-HeartMuLa] Loading MuQ-MuLan model...")
            model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
            _muq_model = model.float().eval()  # from_pretrained loads on CPU
            _muq_loaded_choice = "cpu"

        if _muq_loaded_choice != device_choice:
            target = _resolve_device(device_choice)
            print(f"[JK-HeartMuLa] Moving MuQ-MuLan to {target} ({device_choice})...")
            _muq_model = _muq_model.to(target)
            _muq_loaded_choice = device_choice

        dev = next(_muq_model.parameters()).device
        print(f"[JK-HeartMuLa] MuQ-MuLan ready on {dev}")
        return _muq_model


class JKHeartMuLaMuQModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="JKHeartMuLaMuQModelLoader",
            display_name="HeartMuLa MuQ Model Loader",
            category="JK-HeartMuLa",
            inputs=[
                io.Combo.Input(
                    "device",
                    options=["cpu", "gpu"],
                    default="cpu",
                    tooltip="Where to run MuQ-MuLan for style embedding. 'cpu' uses ~2.5 GB "
                            "system RAM (no VRAM). 'gpu' is faster but keeps ~2.5 GB VRAM "
                            "resident (not freed between runs), which competes with generation.",
                ),
            ],
            outputs=[MUQ_TYPE.Output(display_name="muq_model")],
        )

    @classmethod
    def fingerprint_inputs(cls, device="cpu") -> str:
        # Singleton, but must re-run (and move the model) when the device changes.
        return f"muq_singleton:{device}"

    @classmethod
    def execute(cls, device="cpu") -> io.NodeOutput:
        return io.NodeOutput(_get_muq_model(device))
