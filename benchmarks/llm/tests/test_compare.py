"""The `benchmark compare` report: one row per model present in >=2 of the
four (prompt_variant, addressing_mode) cells, paired delta/p-value between
segmentation/segment_ids and detection/timestamps.
"""
from __future__ import annotations

from benchmark import corpus
from benchmark.report import compare as compare_mod
from benchmark.storage import append_jsonl

from tests.test_addressing_mode import CALL_TEMPLATE, SEGMENTS


def test_compare_with_no_data_at_all(tmp_path, minimal_cfg, pricing_snapshot):
    out = tmp_path / "comparison.md"
    compare_mod.render(
        cfg=minimal_cfg, episodes=[], calls_path=tmp_path / "calls.jsonl",
        pricing_snapshot=pricing_snapshot, output_path=out,
    )
    assert "No benchmark data yet" in out.read_text()


def test_compare_pairs_segmentation_segment_ids_against_detection_timestamps(
    tmp_path, minimal_cfg, pricing_snapshot, write_corpus_episode,
):
    ep_dir = write_corpus_episode(tmp_path / "corpus", segments=SEGMENTS)
    ep = corpus.load_episode(ep_dir)
    calls_path = tmp_path / "calls.jsonl"
    append_jsonl(calls_path, {
        **CALL_TEMPLATE, "call_id": "c1", "episode_id": ep.ep_id,
        "prompt_variant": "detection", "addressing_mode": "timestamps",
        "parsed_ads": [{"start_time": 0.0, "end_time": 30.0}],
    })
    append_jsonl(calls_path, {
        **CALL_TEMPLATE, "call_id": "c2", "episode_id": ep.ep_id,
        "prompt_variant": "segmentation", "addressing_mode": "segment_ids",
        "parsed_ads": [{"start": 0.0, "end": 30.0}],
    })
    # Present in only one cell: must not appear in the comparison at all.
    append_jsonl(calls_path, {
        **CALL_TEMPLATE, "call_id": "c3", "episode_id": ep.ep_id,
        "model": "solo-model", "prompt_variant": "segmentation", "addressing_mode": "timestamps",
        "parsed_ads": [{"start_time": 0.0, "end_time": 30.0}],
    })

    out = tmp_path / "comparison.md"
    compare_mod.render(
        cfg=minimal_cfg, episodes=[ep], calls_path=calls_path,
        pricing_snapshot=pricing_snapshot, output_path=out,
    )
    text = out.read_text()

    assert "detection/timestamps" in text
    assert "segmentation/segment_ids" in text
    assert "solo-model" not in text

    m1_rows = [line for line in text.splitlines() if line.startswith("| `m1`")]
    assert len(m1_rows) == 1, text
    # Only one shared episode between the two paired cells -> p-value n/a,
    # but a delta is still printed.
    assert "n/a" in m1_rows[0]

    assert "## Episode sets per cell" in text
    assert ep.ep_id in text


def test_compare_model_in_only_one_cell_is_omitted(
    tmp_path, minimal_cfg, pricing_snapshot, write_corpus_episode,
):
    ep_dir = write_corpus_episode(tmp_path / "corpus", segments=SEGMENTS)
    ep = corpus.load_episode(ep_dir)
    calls_path = tmp_path / "calls.jsonl"
    append_jsonl(calls_path, {
        **CALL_TEMPLATE, "call_id": "c1", "episode_id": ep.ep_id,
        "prompt_variant": "detection", "addressing_mode": "timestamps",
        "parsed_ads": [{"start_time": 0.0, "end_time": 30.0}],
    })

    out = tmp_path / "comparison.md"
    compare_mod.render(
        cfg=minimal_cfg, episodes=[ep], calls_path=calls_path,
        pricing_snapshot=pricing_snapshot, output_path=out,
    )
    text = out.read_text()
    assert "`m1`" not in text
