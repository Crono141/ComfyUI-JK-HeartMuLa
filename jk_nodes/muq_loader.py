"""HeartMuLa MuQ Model Loader.

Lazy-loads OpenMuQ/MuQ-MuLan-large as a module-level singleton so it is only
loaded once per ComfyUI session. A `device` switch chooses where it runs:
  - "cpu"  -> ~2.5 GB system RAM, no VRAM (default).
  - "gpu"  -> faster style embedding.
Style Embed runs the embedding on whichever device the model is loaded on, and
(optionally) offloads it back to CPU afterwards to free VRAM for generation --
see ``offload_muq`` / ``ensure_on_desired_device`` below.
"""

import threading

import comfy.model_management
import torch
from comfy_api.latest import io

# Custom socket type for the loaded MuQ-MuLan model. Matches the input type on
# HeartMuLa Style Embed.
MUQ_TYPE = io.Custom("JKHEARTMULA_MUQ")

_muq_model = None
_muq_desired_choice = "cpu"  # the device the user selected on the loader
_muq_lock = threading.Lock()


def _resolve_device(device_choice):
    if device_choice == "gpu":
        return comfy.model_management.get_torch_device()
    return torch.device("cpu")


def _current_device():
    return next(_muq_model.parameters()).device


def _get_muq_model(device_choice):
    """Return the process-wide MuQ-MuLan singleton on the requested device."""
    global _muq_model, _muq_desired_choice
    with _muq_lock:
        if _muq_model is None:
            from muq import MuQMuLan

            print("[JK-HeartMuLa] Loading MuQ-MuLan model...")
            model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
            _muq_model = model.float().eval()  # from_pretrained loads on CPU

        _muq_desired_choice = device_choice
        target = _resolve_device(device_choice)
        if _current_device() != target:
            print(f"[JK-HeartMuLa] Moving MuQ-MuLan to {target} ({device_choice})...")
            _muq_model = _muq_model.to(target)
        print(f"[JK-HeartMuLa] MuQ-MuLan ready on {_current_device()}")
        return _muq_model


def ensure_on_desired_device():
    """Move the singleton back onto the loader's selected device (used by Style
    Embed before embedding, in case it was previously offloaded to CPU). No-op if
    nothing is loaded."""
    with _muq_lock:
        if _muq_model is None:
            return None
        target = _resolve_device(_muq_desired_choice)
        if _current_device() != target:
            print(f"[JK-HeartMuLa] Restoring MuQ-MuLan to {target} for embedding...")
            _muq_model.to(target)
        return _muq_model


def offload_muq():
    """Move the singleton to CPU and free its VRAM. Safe to call repeatedly; the
    model stays cached in RAM so it doesn't reload from disk."""
    with _muq_lock:
        if _muq_model is not None and _current_device().type != "cpu":
            print("[JK-HeartMuLa] Offloading MuQ-MuLan to CPU (freeing VRAM)...")
            _muq_model.to("cpu")
    comfy.model_management.soft_empty_cache()


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
                            "system RAM (no VRAM). 'gpu' is faster; pair it with Style Embed's "
                            "'free_vram_after' so the VRAM is released before generation.",
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
