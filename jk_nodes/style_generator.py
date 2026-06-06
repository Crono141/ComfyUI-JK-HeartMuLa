"""HeartMuLa Music Generator (with optional style transfer).

This replaces ComfyUI-HeartMuLa's original "Music Generator": same HEARTMULA_MODEL
input, same widgets, same HEARTMULA_TOKENS output -- so it chains straight into
the "HeartMuLa Audio Decoder". The one addition is an optional ``cmuq`` input
(from HeartMuLa Style Embed). With ``cmuq`` unconnected it behaves exactly like
the stock generator.

How the style conditioning works (resolved from heartlib source, not guessed):
``model.preprocess()`` builds a ``muq_embed`` vector that heartlib hardcodes to
zeros for tag-only generation, injected at ``muq_idx`` (right after the tags).
``generate_tokens()`` passes it through as ``continuous_segments`` to
``model.generate_frame()``, which projects it via ``muq_linear`` and writes it
into the prompt's hidden state. To do style transfer we simply overwrite that
zero vector with the reference-audio embedding before generation. The model
handles classifier-free guidance internally (it overwrites the unconditional
half of the batch with its own learned embedding), so we just fill every batch
row with the same embedding.
"""

import torch

import comfy.model_management
import comfy.utils
from comfy_api.latest import io

# Custom socket types. The HEARTMULA_* strings match ComfyUI-HeartMuLa's loaders
# (intentional -- only node ids must be unique, not socket types). JKHEARTMULA_CMUQ
# matches the output of HeartMuLa Style Embed.
HEARTMULA_MODEL = io.Custom("HEARTMULA_MODEL")
HEARTMULA_TOKENS = io.Custom("HEARTMULA_TOKENS")
CMUQ_TYPE = io.Custom("JKHEARTMULA_CMUQ")


class JKHeartMuLaMusicGenerator(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="JKHeartMuLaMusicGenerator",
            display_name="HeartMuLa Music Generator",
            category="JK-HeartMuLa",
            inputs=[
                HEARTMULA_MODEL.Input("model"),
                io.String.Input("lyrics", multiline=True, default=""),
                io.String.Input("tags", multiline=True, default=""),
                io.Float.Input("duration_seconds", default=30.0, min=1.0, max=300.0),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Float.Input("temperature", default=1.0, min=0.1, max=2.0),
                io.Int.Input("top_k", default=50, min=1, max=1000),
                io.Float.Input("cfg_scale", default=1.5, min=0.1, max=5.0),
                # Leave unconnected for standard tag-only generation.
                CMUQ_TYPE.Input("cmuq", optional=True),
            ],
            outputs=[HEARTMULA_TOKENS.Output(display_name="tokens")],
        )

    @classmethod
    def execute(cls, model, lyrics, tags, duration_seconds, seed, temperature,
                top_k, cfg_scale, cmuq=None) -> io.NodeOutput:
        torch.manual_seed(seed)
        try:
            torch.set_float32_matmul_precision("high")
            torch._dynamo.reset()
        except Exception:
            pass

        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        device = comfy.model_management.get_torch_device()

        cleaned_tags = ",".join([t.strip() for t in tags.split(",") if t.strip()])

        style_transfer = cmuq is not None
        print(f"[JK-HeartMuLa] Generating with seed={seed}, "
              f"style_transfer={'yes' if style_transfer else 'no'}")

        try:
            inputs = {"lyrics": lyrics, "tags": cleaned_tags}
            processed_inputs = model.preprocess(inputs, cfg_scale)
            max_audio_length_ms = int(duration_seconds * 1000)

            # Inject the reference-audio style embedding, replacing heartlib's
            # zeroed muq_embed. Match the existing tensor's batch/dtype so the
            # downstream device-move loop and CFG handling are unchanged.
            if style_transfer:
                target = processed_inputs["muq_embed"]  # [bs, muq_dim]
                emb = cmuq.detach().reshape(1, -1).to(dtype=target.dtype)
                if emb.shape[-1] != target.shape[-1]:
                    raise ValueError(
                        f"cmuq dim {emb.shape[-1]} != model muq_dim "
                        f"{target.shape[-1]}")
                processed_inputs["muq_embed"] = emb.expand_as(target).contiguous()

            print(f"Moving HeartMuLa Generator to {device}...")
            model.model.to(device)
            torch.cuda.synchronize()

            model.setup_caches(max_batch_size=processed_inputs["tokens"].shape[0])
            model.ensure_compiled()

            for k, v in processed_inputs.items():
                if isinstance(v, torch.Tensor):
                    processed_inputs[k] = v.to(device)
                    if v.is_floating_point():
                        processed_inputs[k] = processed_inputs[k].to(dtype=model.dtype)

            gen_steps = int(duration_seconds * 1000 // 80)
            pbar = comfy.utils.ProgressBar(gen_steps)

            def gen_callback(step):
                pbar.update(1)

            with torch.no_grad():
                tokens = model.generate_tokens(
                    processed_inputs,
                    max_audio_length_ms=max_audio_length_ms,
                    temperature=temperature,
                    topk=top_k,
                    cfg_scale=cfg_scale,
                    callback=gen_callback,
                )

            torch.cuda.synchronize()
            return io.NodeOutput(tokens)
        finally:
            model.model.to("cpu")
            torch.cuda.synchronize()
            comfy.model_management.soft_empty_cache()
