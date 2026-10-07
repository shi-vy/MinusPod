"""Segmentation prompt variant: frozen system prompt, user prompt build, and
response parsing, kept alongside the detection path production ships."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from config import repair_segment_category  # type: ignore[import-not-found]

from . import parsing

logger = logging.getLogger(__name__)

PROMPT_VARIANTS = ("detection", "segmentation")
DEFAULT_VARIANT = "detection"
ADDRESSING_MODES = ("timestamps", "segment_ids")
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


def _to_int_or_none(value) -> int | None:
    """value as an int if it is integer-valued (e.g. 1, 1.0, "1.0"), else None."""
    try:
        f = float(value)
        return int(f) if f == int(f) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _normalize_direct_method(method: str | None) -> str | None:
    """A raw JSON-wrapping method that carries ads collapses to the one
    compliance label the segmentation prompt's object shape earns."""
    if method in ("json_object_segments_key", "json_array_direct"):
        return "segmentation_object_direct"
    return method


def parse_segmentation_response(
    response_text: str, addressing_mode: str, id_segments: list[dict] | None,
) -> tuple[list[dict], str | None, bool]:
    raw, method = parsing.extract_json_ads_array(response_text)
    if raw is None:
        return [], method, False

    kept = [
        entry for entry in raw
        if isinstance(entry, dict) and repair_segment_category(entry.get("category")) in PROMO_CATEGORIES
    ]
    if not kept:
        # A valid all-main_content answer: no promo entries to judge the id
        # contract against, so score it as full compliance for the mode in use.
        no_promo_method = "segment_id_direct" if addressing_mode == "segment_ids" else "segmentation_object_direct"
        return [], no_promo_method, False

    if addressing_mode == "segment_ids":
        id_entries = []
        skipped = 0
        for entry in kept:
            start_id, end_id = _to_int_or_none(entry.get("start")), _to_int_or_none(entry.get("end"))
            if start_id is not None and end_id is not None:
                id_entry = {k: v for k, v in entry.items() if k not in ("start", "end")}
                id_entry["start_id"] = start_id
                id_entry["end_id"] = end_id
                id_entries.append(id_entry)
            else:
                skipped += 1
        if id_entries:
            if skipped:
                logger.warning(
                    "segmentation parse: ID-mode response mixed formats: "
                    "skipped %d entr%s without integer-valued start/end", skipped, "y" if skipped == 1 else "ies",
                )
            resolved = parsing.resolve_segment_id_ads(id_entries, id_segments or [])
            return resolved, "segment_id_direct", False
        # Model ignored the id contract but gave usable floats; mirrors
        # runner._parse_id_response's fallback-to-timestamps semantics.
        parsed = parsing.parse_ads_from_response(json.dumps(kept)) or []
        return list(parsed), _normalize_direct_method(method), True

    parsed = parsing.parse_ads_from_response(json.dumps(kept)) or []
    return list(parsed), _normalize_direct_method(method), False
