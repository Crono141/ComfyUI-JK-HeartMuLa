"""ComfyUI-JK-HeartMuLa.

A fork of BobRandomNumber/ComfyUI-HeartMuLa (Apache-2.0) that adds MuQ-MuLan
reference-audio style transfer. All node ids are uniquely prefixed so this pack
can be installed alongside the original ComfyUI-HeartMuLa without collisions.

- Bob's nodes are reused from nodes.py, rebranded with `JKHeartMuLa*` ids and the
  `JK-HeartMuLa` category.
- His Music Generator is replaced by jk_nodes/style_generator.py (adds an optional
  `cmuq` style-embedding input; identical behavior when it is unconnected).
- New nodes (HML MuQ Model Loader, HML Style Embed) live in jk_nodes/.
"""

import os
import platform

import server
from aiohttp import web
from comfy_api.latest import ComfyExtension, io

# Bob's nodes (rebranded ids/category live inside nodes.py). His original
# HeartMuLaMusicGenerator is intentionally not imported -- it is replaced below.
from .nodes import (
    HeartMuLaLoader,
    HeartMuLaCodecLoader,
    HeartMuLaAudioDecoder,
    HeartMuLaTranscriptionLoader,
    HeartMuLaLyricsTranscriber,
    HeartMuLaPostProcessor,
)

# Our additions.
from .jk_nodes.muq_loader import JKHeartMuLaMuQModelLoader
from .jk_nodes.style_embed import JKHeartMuLaStyleEmbed
from .jk_nodes.style_generator import JKHeartMuLaMusicGenerator
from .jk_nodes.tags_builder import JKHeartMuLaTagsBuilder


class JKHeartMuLaExtension(ComfyExtension):
    def __init__(self):
        super().__init__()
        self._register_routes()

    def _register_routes(self):
        # Namespaced route (/jkheartmula/...) so it does not collide with the
        # original ComfyUI-HeartMuLa pack's /heartmula/browse route.
        @server.PromptServer.instance.routes.get("/jkheartmula/browse")
        async def browse_directory(request):
            path = request.query.get("path", "")
            is_windows = platform.system() == "Windows"

            if not path or path == "root":
                if is_windows:
                    import string
                    drives = []
                    for letter in string.ascii_uppercase:
                        drive = f"{letter}:\\"
                        if os.path.exists(drive):
                            drives.append(drive)
                    return web.json_response({
                        "current_path": "root",
                        "dirs": drives,
                        "parent": None,
                        "is_windows_root": True
                    })
                else:
                    path = "/"

            if not os.path.exists(path):
                return web.json_response({"error": f"Path not found: {path}"}, status=404)

            try:
                items = os.listdir(path)
                dirs = []
                for d in items:
                    try:
                        full_item_path = os.path.join(path, d)
                        if os.path.isdir(full_item_path):
                            dirs.append(d)
                    except (PermissionError, OSError):
                        continue

                parent = os.path.abspath(os.path.join(path, ".."))
                if is_windows and (parent == path or len(path) <= 3):
                    parent = "root"

                return web.json_response({
                    "current_path": os.path.abspath(path),
                    "dirs": sorted(dirs),
                    "parent": parent,
                    "is_windows_root": False
                })
            except Exception as e:
                return web.json_response({"error": str(e)}, status=500)

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            HeartMuLaLoader,
            HeartMuLaCodecLoader,
            JKHeartMuLaTagsBuilder,
            JKHeartMuLaMusicGenerator,   # replaces HeartMuLa's Music Generator
            HeartMuLaAudioDecoder,
            HeartMuLaTranscriptionLoader,
            HeartMuLaLyricsTranscriber,
            HeartMuLaPostProcessor,
            JKHeartMuLaMuQModelLoader,
            JKHeartMuLaStyleEmbed,
        ]


async def comfy_entrypoint() -> JKHeartMuLaExtension:
    return JKHeartMuLaExtension()


WEB_DIRECTORY = "js"
