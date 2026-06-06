"""HeartMuLa Tag Builder.

A V3 port of the RT-HeartMuLa "HeartMuLa Tags Builder": per-category free-text /
dropdown inputs that are lowercased, de-duplicated (first occurrence wins), and
comma-joined into a single tag string for the Music Generator's `tags` input.

The only deliberate change from the RT version is that `additional_tags` is a
multiline text box (instead of a single line that overflows).
"""

from datetime import datetime

from comfy_api.latest import io

ORANGE = "\033[38;5;208m"
RESET = "\033[0m"

GENRE_TAGS = [
    "pop", "rock", "jazz", "electronic", "classical", "hip-hop", "r&b",
    "country", "reggae", "blues", "folk", "metal", "punk", "disco", "funk",
    "soul", "indie", "edm", "house", "techno", "ambient", "lo-fi", "k-pop",
    "j-pop", "c-pop", "latin", "reggaeton", "bossa nova",
]
MOOD_TAGS = [
    "happy", "sad", "energetic", "calm", "romantic", "melancholic",
    "uplifting", "dark", "dreamy", "aggressive", "peaceful", "nostalgic",
    "hopeful", "mysterious", "playful", "epic", "chill", "intense",
]
INSTRUMENT_TAGS = [
    "piano", "guitar", "acoustic guitar", "electric guitar", "bass", "drums",
    "violin", "cello", "saxophone", "trumpet", "flute", "synth", "strings",
    "orchestra", "choir", "808", "harp", "organ", "ukulele", "harmonica",
]
TEMPO_TAGS = [
    "none", "slow", "moderate", "fast", "very fast", "ballad", "uptempo",
    "mid-tempo", "groovy",
]
VOCAL_TAGS = [
    "male vocal", "female vocal", "duet", "choir", "rap", "whisper",
    "powerful vocals", "falsetto", "soprano", "baritone", "tenor", "alto",
    "harmonies", "a cappella",
]
ERA_TAGS = [
    "none", "80s", "90s", "2000s", "modern", "retro", "vintage", "70s", "60s",
    "futuristic",
]
PRODUCTION_TAGS = [
    "cinematic", "lo-fi", "hi-fi", "acoustic", "electronic", "live", "studio",
    "raw", "polished", "minimalist", "layered", "atmospheric",
]


def _tooltip(name, tags):
    return f"{name} tags (comma-separated). Available: {', '.join(tags)}"


class JKHeartMuLaTagsBuilder(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="JKHeartMuLaTagsBuilder",
            display_name="HeartMuLa Tag Builder",
            category="JK-HeartMuLa",
            inputs=[
                io.String.Input("genre", default="", placeholder="pop, rock, ...",
                                tooltip=_tooltip("Genre", GENRE_TAGS)),
                io.String.Input("mood", default="", placeholder="happy, dreamy, ...",
                                tooltip=_tooltip("Mood", MOOD_TAGS)),
                io.String.Input("instrument", default="", placeholder="piano, guitar, ...",
                                tooltip=_tooltip("Instrument", INSTRUMENT_TAGS)),
                io.Combo.Input("tempo", options=TEMPO_TAGS, default="none"),
                io.String.Input("vocal", default="", placeholder="female vocal, harmonies, ...",
                                tooltip=_tooltip("Vocal", VOCAL_TAGS)),
                io.Combo.Input("era", options=ERA_TAGS, default="none"),
                io.String.Input("production", default="", placeholder="cinematic, lo-fi, ...",
                                tooltip=_tooltip("Production", PRODUCTION_TAGS)),
                # Multiline (the one change from the RT builder).
                io.String.Input("additional_tags", default="", multiline=True, optional=True,
                                placeholder="anything not covered above",
                                tooltip="Free-text comma-separated tags. Use for anything not "
                                        "in the dropdowns or per-category lists."),
            ],
            outputs=[io.String.Output(display_name="tags")],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        # Force a re-run on every Queue Prompt so the "process started" banner
        # always appears (matches the RT builder's IS_CHANGED behavior).
        return float("NaN")

    @classmethod
    def execute(cls, genre, mood, instrument, tempo, vocal, era, production,
                additional_tags="") -> io.NodeOutput:
        start_clock = datetime.now().strftime("%I:%M:%S %p")
        print(f"\n{ORANGE}--------------------process started "
              f"[@ {start_clock}]--------------------{RESET}")

        # Per-category inputs in spec order. Each item is either a raw user STRING
        # (comma-separated, possibly multi-tag) or a single dropdown pick.
        sources = [genre, mood, instrument, tempo, vocal, era, production, additional_tags]

        # Paper compliance: tags lowercase, comma-joined, no "none", deduped
        # (first occurrence wins).
        seen = set()
        clean_tags = []
        for src in sources:
            for raw in str(src).split(","):
                t = raw.lower().strip()
                if not t or t == "none" or t in seen:
                    continue
                seen.add(t)
                clean_tags.append(t)

        formatted_tags = ", ".join(clean_tags)
        return io.NodeOutput(formatted_tags)
