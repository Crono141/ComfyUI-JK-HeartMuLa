"""HeartMuLa Tag Builder.

Builds a single, generator-ready tag string from a set of music-description
fields. Values are split on commas, lowercased, trimmed, de-duplicated (first
occurrence wins), with blanks and "none" dropped, then comma-joined -- the form
HeartMuLa's tag conditioning expects.

Original implementation. The general idea of a category-based music tag builder
was inspired by RT-HeartMuLa, but none of its code or tag data is used here.
"""

from comfy_api.latest import io

# Suggestion lists surfaced in tooltips. The free-text fields accept any
# comma-separated values; these are only hints, not a fixed vocabulary.
GENRE_SUGGESTIONS = [
    "pop", "rock", "hip hop", "electronic", "jazz", "classical", "r&b", "soul",
    "funk", "disco", "house", "techno", "trance", "drum and bass", "ambient",
    "lo-fi", "synthwave", "metal", "punk", "indie", "folk", "country", "blues",
    "gospel", "reggae", "latin", "afrobeat", "k-pop", "orchestral", "cinematic",
]
MOOD_SUGGESTIONS = [
    "uplifting", "melancholic", "energetic", "mellow", "dreamy", "dark",
    "euphoric", "nostalgic", "tense", "triumphant", "romantic", "somber",
    "playful", "hypnotic", "anthemic", "serene", "gritty", "bittersweet",
]
INSTRUMENT_SUGGESTIONS = [
    "acoustic guitar", "electric guitar", "piano", "synthesizer", "bass guitar",
    "drum kit", "808 bass", "strings", "brass", "saxophone", "violin", "cello",
    "flute", "organ", "electric piano", "pads", "arpeggiated synth",
    "hand percussion", "choir", "harp",
]
VOCAL_SUGGESTIONS = [
    "male vocals", "female vocals", "duet", "group vocals", "choir", "rap",
    "spoken word", "falsetto", "belting", "breathy", "layered harmonies",
    "vocoder", "instrumental",
]
PRODUCTION_SUGGESTIONS = [
    "polished", "raw", "lo-fi", "hi-fi", "warm", "punchy", "spacious",
    "compressed", "vintage tape", "reverb-heavy", "dry", "wide stereo",
    "minimal", "dense",
]

# Single-pick dropdowns. "none" is treated as "leave unset".
TEMPO_OPTIONS = [
    "none", "very slow", "slow", "mid-tempo", "upbeat", "fast", "very fast",
    "half-time", "double-time", "driving",
]
ERA_OPTIONS = [
    "none", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s",
    "contemporary", "retro", "futuristic",
]


def _hint(label, words):
    return f"{label} — type any comma-separated tags. Suggestions: {', '.join(words)}"


class JKHeartMuLaTagsBuilder(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="JKHeartMuLaTagsBuilder",
            display_name="HeartMuLa Tag Builder",
            category="JK-HeartMuLa",
            inputs=[
                io.String.Input("genre", default="", placeholder="e.g. synthwave, pop",
                                tooltip=_hint("Genre", GENRE_SUGGESTIONS)),
                io.String.Input("mood", default="", placeholder="e.g. dreamy, euphoric",
                                tooltip=_hint("Mood", MOOD_SUGGESTIONS)),
                io.String.Input("instrument", default="", placeholder="e.g. piano, 808 bass",
                                tooltip=_hint("Instruments", INSTRUMENT_SUGGESTIONS)),
                io.Combo.Input("tempo", options=TEMPO_OPTIONS, default="none",
                               tooltip="Overall tempo feel (single pick)."),
                io.String.Input("vocal", default="", placeholder="e.g. female vocals, layered harmonies",
                                tooltip=_hint("Vocals", VOCAL_SUGGESTIONS)),
                io.Combo.Input("era", options=ERA_OPTIONS, default="none",
                               tooltip="Production era (single pick)."),
                io.String.Input("production", default="", placeholder="e.g. warm, wide stereo",
                                tooltip=_hint("Production", PRODUCTION_SUGGESTIONS)),
                io.String.Input("additional_tags", default="", multiline=True, optional=True,
                                placeholder="anything not covered above, comma-separated",
                                tooltip="Free-form comma-separated tags for anything the fields "
                                        "above don't cover."),
            ],
            outputs=[io.String.Output(display_name="tags")],
        )

    @classmethod
    def execute(cls, genre, mood, instrument, tempo, vocal, era, production,
                additional_tags="") -> io.NodeOutput:
        fields = [genre, mood, instrument, tempo, vocal, era, production, additional_tags]

        seen = set()
        tags = []
        for field in fields:
            for token in str(field).split(","):
                tag = token.strip().lower()
                if not tag or tag == "none" or tag in seen:
                    continue
                seen.add(tag)
                tags.append(tag)

        result = ", ".join(tags)
        print(f"[JK-HeartMuLa] Tag Builder -> {result!r}")
        return io.NodeOutput(result)
