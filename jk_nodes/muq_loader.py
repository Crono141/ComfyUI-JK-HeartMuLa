"""HeartMuLa MuQ Model Loader.

Lazy-loads OpenMuQ/MuQ-MuLan-large on CPU as a module-level singleton so it is
only loaded once per ComfyUI session (~2.5 GB RAM, no VRAM impact). The model is
shared by every Style Embed node downstream.
"""

import threading

from comfy_api.latest import io

# Custom socket type for the loaded MuQ-MuLan model. Matches the string used as
# the input type on HeartMuLa Style Embed.
MUQ_TYPE = io.Custom("JKHEARTMULA_MUQ")

_muq_model = None
_muq_lock = threading.Lock()


def _get_muq_model():
    """Return the process-wide MuQ-MuLan singleton, loading it on first use."""
    global _muq_model
    with _muq_lock:
        if _muq_model is None:
            from muq import MuQMuLan

            print("[JK-HeartMuLa] Loading MuQ-MuLan model on CPU...")
            model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
            _muq_model = model.to("cpu").float().eval()
            print("[JK-HeartMuLa] MuQ-MuLan ready (~2.5 GB RAM)")
        return _muq_model


class JKHeartMuLaMuQModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="JKHeartMuLaMuQModelLoader",
            display_name="HeartMuLa MuQ Model Loader",
            category="JK-HeartMuLa",
            inputs=[],
            outputs=[MUQ_TYPE.Output(display_name="muq_model")],
        )

    @classmethod
    def fingerprint_inputs(cls) -> str:
        # The model is a singleton with no inputs; return a constant so the node
        # never re-executes (equivalent to V1's IS_CHANGED).
        return "muq_singleton"

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        return io.NodeOutput(_get_muq_model())
