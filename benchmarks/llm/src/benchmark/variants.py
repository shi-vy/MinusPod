"""Segmentation prompt variant: frozen system prompt, user prompt build, and
response parsing, kept alongside the detection path production ships."""
from __future__ import annotations

import json
import re
from pathlib import Path

from config import repair_segment_category  # type: ignore[import-not-found]

from . import parsing

PROMPT_VARIANTS = ("detection", "segmentation")
DEFAULT_VARIANT = "detection"
SEGMENTATION_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "segmentation-v1.txt"
# main_content/teaser/transition fail repair_segment_category; intro/outro/recap
# are dropped here too, so merged rows stay comparable to the detection prompt.
PROMO_CATEGORIES = frozenset({"sponsor", "cross_promo", "self_promo", "interaction"})


def validate_variant(name: str) -> None:
    if name not in PROMPT_VARIANTS:
        raise ValueError(f"unknown prompt variant {name!r}; choose from {PROMPT_VARIANTS}")


def finalize_prompt(prompt: str) -> str:
    """Strip, drop trailing per-line whitespace, collapse 3+ newlines to 2."""
    prompt = prompt.strip() if prompt else ""
    if not prompt:
        return ""
    prompt = re.sub(r"[ \t]+\r?\n", "\n", prompt)
    prompt = re.sub(r"(?:\r?\n){3,}", "\n\n", prompt)
    return prompt


def segmentation_system_prompt() -> str:
    return finalize_prompt(SEGMENTATION_PROMPT_PATH.read_text())


# Mirrors the PR's USER_PROMPT_TEMPLATE (ad_detector/prompts.py); the
# benchmark has no audio_context, so callers always pass "".
_USER_PROMPT_TEMPLATE = """
Podcast: {podcast_name}
Episode: {episode_title}
{description_section}

{window_context}

{transcript}

{audio_context}"""


def _build_window_context(
    window_index: int, total_windows: int, window_start: float, window_end: float,
) -> str:
    if total_windows <= 1:
        return ""
    window_context = (
        f"IMPORTANT: This is a partial transcript (window {window_index + 1} of {total_windows}) "
        f"with hard cuts at minutes {window_start/60:.1f} thru {window_end/60:.1f}.\n"
    )
    if window_index > 0 and window_index + 1 < total_windows:
        window_context += (
            "If the first or last segment appears to begin or end outside this "
            "window, append \"continues from previous\" or \"continues in next\" "
            "to the `reason` field.\n"
        )
    elif window_index > 0:
        window_context += (
            "If the first segment appears to begin outside this window, append "
            "\"continues from previous\" to the `reason` field.\n"
        )
    elif window_index + 1 < total_windows:
        window_context += (
            "If the last segment appears to end outside this window, append "
            "\"continues in next\" to the `reason` field.\n"
        )
    return window_context


def format_segmentation_prompt(
    podcast_name: str,
    episode_title: str,
    description_section: str,
    transcript_lines: list[str],
    window_index: int,
    total_windows: int,
    window_start: float,
    window_end: float,
    addressing_mode: str = "timestamps",
) -> str:
    window_context = _build_window_context(window_index, total_windows, window_start, window_end)
    transcript = (
        "Transcript with timestamps:\n"
        if addressing_mode == "timestamps"
        else "Transcript with indexes:\n"
    )
    transcript += "\n".join(transcript_lines)
    return finalize_prompt(
        _USER_PROMPT_TEMPLATE.format(
            podcast_name=podcast_name,
            episode_title=episode_title,
            description_section=description_section,
            window_context=window_context,
            transcript=transcript,
            audio_context="",
        )
    )


def _is_int_valued(value) -> bool:
    try:
        return float(value) == int(float(value))
    except (TypeError, ValueError):
        return False


def parse_segmentation_response(
    response_text: str, addressing_mode: str, id_segments: list[dict] | None,
) -> tuple[list[dict], str, bool]:
    raw, method = parsing.extract_json_ads_array(response_text)
    if raw is None:
        return [], method or "none", False

    kept = [
        entry for entry in raw
        if isinstance(entry, dict) and repair_segment_category(entry.get("category")) in PROMO_CATEGORIES
    ]
    if not kept:
        return [], method, False

    if addressing_mode == "segment_ids":
        id_entries = []
        for entry in kept:
            start, end = entry.get("start"), entry.get("end")
            if _is_int_valued(start) and _is_int_valued(end):
                id_entry = {k: v for k, v in entry.items() if k not in ("start", "end")}
                id_entry["start_id"] = int(start)
                id_entry["end_id"] = int(end)
                id_entries.append(id_entry)
        if id_entries:
            resolved = parsing.resolve_segment_id_ads(id_entries, id_segments or [])
            return resolved, "segment_id_direct", False
        # Model ignored the id contract but gave usable floats; mirrors
        # runner._parse_id_response's fallback-to-timestamps semantics.
        parsed = parsing.parse_ads_from_response(json.dumps(kept)) or []
        fallback_method = (
            "segmentation_object_direct" if method in ("json_object_segments_key", "json_array_direct")
            else method
        )
        return list(parsed), fallback_method, True

    parsed = parsing.parse_ads_from_response(json.dumps(kept)) or []
    out_method = (
        "segmentation_object_direct" if method in ("json_object_segments_key", "json_array_direct")
        else method
    )
    return list(parsed), out_method, False
