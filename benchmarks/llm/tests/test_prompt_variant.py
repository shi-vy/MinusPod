"""Threading prompt_variant through the CLI, runner, and call records.

Mirrors tests/test_addressing_mode.py's fixture style. No benchmark calls here.
"""
from __future__ import annotations

import asyncio

from typer.testing import CliRunner

from benchmark import cli, corpus, runner
from benchmark.llm import LLMResponse
from benchmark.storage import append_jsonl, read_jsonl

from tests.test_cli import write_minimal_config


# --- detection variant unchanged by default ----------------------------------

def test_detection_default_matches_explicit_kwarg_and_pre_change_behavior(make_episode, minimal_cfg):
    ep = make_episode()
    default_prompt = runner._build_user_prompt(ep, ep.windows[0], total_windows=1)
    explicit_prompt = runner._build_user_prompt(
        ep, ep.windows[0], total_windows=1, prompt_variant="detection",
    )
    assert default_prompt == explicit_prompt

    default_hashes = runner.precompute_prompt_hashes(minimal_cfg, [ep], system_prompt="S")
    explicit_hashes = runner.precompute_prompt_hashes(
        minimal_cfg, [ep], system_prompt="S", prompt_variant="detection",
    )
    assert default_hashes == explicit_hashes


# --- segmentation variant: user prompt + hash differ -------------------------

def test_segmentation_prompt_and_hash_differ_from_detection(tmp_path, write_corpus_episode, minimal_cfg):
    ep_dir = write_corpus_episode(tmp_path)
    ep = corpus.load_episode(ep_dir)

    detection_hashes = runner.precompute_prompt_hashes(minimal_cfg, [ep], system_prompt="S")
    segmentation_hashes = runner.precompute_prompt_hashes(
        minimal_cfg, [ep], system_prompt="S", prompt_variant="segmentation",
    )
    assert detection_hashes[("m1", ep.ep_id, 0, 0)] != segmentation_hashes[("m1", ep.ep_id, 0, 0)]


def test_segmentation_prompt_id_mode_header_says_indexes(tmp_path, write_corpus_episode):
    ep_dir = write_corpus_episode(tmp_path)
    ep = corpus.load_episode(ep_dir)
    prompt = runner._build_user_prompt(
        ep, ep.windows[0], total_windows=1,
        addressing_mode="segment_ids", prompt_variant="segmentation",
        id_segments=corpus.stamp_id_windows(ep)[0],
    )
    assert "Transcript with indexes:" in prompt


# --- reconstruct_user_prompt: legacy default ---------------------------------

def test_reconstruct_user_prompt_missing_field_defaults_to_detection(tmp_path, write_corpus_episode):
    ep_dir = write_corpus_episode(tmp_path)
    ep = corpus.load_episode(ep_dir)
    rebuilt = runner.reconstruct_user_prompt({"episode_id": ep.ep_id, "window_index": 0}, corpus_dir=tmp_path)
    assert rebuilt == runner._build_user_prompt(ep, ep.windows[0], total_windows=1, prompt_variant="detection")


# --- run() end-to-end: record field + derive_episode_results keying ---------

def test_run_writes_prompt_variant_on_record(tmp_path, minimal_cfg, make_episode, pricing_snapshot, monkeypatch):
    async def fake_call(**kwargs):
        return LLMResponse(
            text='[{"start_time": 0.0, "end_time": 30.0}]',
            input_tokens=100, output_tokens=10,
            json_format_used="native", underlying_provider="openrouter", stop_reason="stop",
        )

    monkeypatch.setattr(runner.llm, "call_with_retry", fake_call)
    ep = make_episode(n_windows=1)
    paths = runner.RunPaths.for_root(tmp_path)
    asyncio.run(runner.run(
        minimal_cfg, [ep], paths=paths, pricing_snapshot=pricing_snapshot, system_prompt="S",
        prompt_variant="segmentation",
    ))
    records = list(read_jsonl(paths.calls_jsonl))
    assert records
    assert all(r["prompt_variant"] == "segmentation" for r in records)


def test_derive_episode_results_keeps_rows_for_two_variants(tmp_path, minimal_cfg, make_episode):
    ep = make_episode(n_windows=1)
    calls_path = tmp_path / "raw" / "calls.jsonl"
    paths = runner.RunPaths.for_root(tmp_path)
    base = {
        "schema_version": 2, "model": "m1", "episode_id": ep.ep_id, "trial": 0,
        "window_index": 0, "input_tokens": 10, "output_tokens": 5, "response_time_ms": 1,
        "parsed_ads": [], "error": None,
    }
    append_jsonl(calls_path, {**base, "addressing_mode": "timestamps", "prompt_variant": "detection"})
    append_jsonl(calls_path, {**base, "addressing_mode": "timestamps", "prompt_variant": "segmentation"})

    runner.derive_episode_results(minimal_cfg, [ep], paths=paths)
    results = list(read_jsonl(paths.episode_results_jsonl))
    assert len(results) == 2
    variants_seen = {r["prompt_variant"] for r in results}
    assert variants_seen == {"detection", "segmentation"}
    for r in results:
        assert r["addressing_mode"] == "timestamps"


def test_derive_episode_results_legacy_record_defaults(tmp_path, minimal_cfg, make_episode):
    ep = make_episode(n_windows=1)
    calls_path = tmp_path / "raw" / "calls.jsonl"
    paths = runner.RunPaths.for_root(tmp_path)
    append_jsonl(calls_path, {
        "schema_version": 2, "model": "m1", "episode_id": ep.ep_id, "trial": 0,
        "window_index": 0, "input_tokens": 10, "output_tokens": 5, "response_time_ms": 1,
        "parsed_ads": [], "error": None,
    })  # no addressing_mode or prompt_variant key, as every call before this feature existed

    runner.derive_episode_results(minimal_cfg, [ep], paths=paths)
    results = list(read_jsonl(paths.episode_results_jsonl))
    assert len(results) == 1
    assert results[0]["addressing_mode"] == "timestamps"
    assert results[0]["prompt_variant"] == "detection"


# --- CLI: --model / --episode filters ----------------------------------------

def test_run_dry_run_model_filter_narrows_work_list(tmp_path, monkeypatch):
    cli_runner = CliRunner()
    cfg_path = write_minimal_config(tmp_path)
    corpus_dir = tmp_path / "data" / "corpus"
    monkeypatch.chdir(tmp_path)
    _make_corpus_episode(corpus_dir, "ep-a")
    _make_corpus_episode(corpus_dir, "ep-b")

    result = cli_runner.invoke(cli.app, ["run", "--config", str(cfg_path), "--dry-run"])
    assert result.exit_code == 0
    assert "10 calls would execute" in result.stdout  # 1 model x 2 episodes x 1 window x 5 trials

    result = cli_runner.invoke(
        cli.app, ["run", "--config", str(cfg_path), "--dry-run", "--episode", "ep-a"],
    )
    assert result.exit_code == 0
    assert "5 calls would execute" in result.stdout  # filtered to 1 episode


def test_run_unknown_model_filter_exits_2(tmp_path, monkeypatch):
    cli_runner = CliRunner()
    cfg_path = write_minimal_config(tmp_path)
    corpus_dir = tmp_path / "data" / "corpus"
    monkeypatch.chdir(tmp_path)
    _make_corpus_episode(corpus_dir, "ep-a")

    result = cli_runner.invoke(
        cli.app, ["run", "--config", str(cfg_path), "--dry-run", "--model", "bogus-model"],
    )
    assert result.exit_code == 2
    assert "bogus-model" in result.output


def test_run_unknown_episode_filter_exits_2(tmp_path, monkeypatch):
    cli_runner = CliRunner()
    cfg_path = write_minimal_config(tmp_path)
    corpus_dir = tmp_path / "data" / "corpus"
    monkeypatch.chdir(tmp_path)
    _make_corpus_episode(corpus_dir, "ep-a")

    result = cli_runner.invoke(
        cli.app, ["run", "--config", str(cfg_path), "--dry-run", "--episode", "ep-bogus"],
    )
    assert result.exit_code == 2
    assert "ep-bogus" in result.output


# --- CLI: segmentation + --snapshot rejected ---------------------------------

def test_run_segmentation_with_snapshot_exits_2(tmp_path, monkeypatch):
    cli_runner = CliRunner()
    cfg_path = write_minimal_config(tmp_path)
    corpus_dir = tmp_path / "data" / "corpus"
    monkeypatch.chdir(tmp_path)
    _make_corpus_episode(corpus_dir, "ep-a")
    snapshot_path = tmp_path / "snap.txt"
    snapshot_path.write_text("frozen prompt")

    result = cli_runner.invoke(
        cli.app,
        ["run", "--config", str(cfg_path), "--dry-run", "--prompt-variant", "segmentation", "--snapshot", str(snapshot_path)],
    )
    assert result.exit_code == 2
    assert "segmentation variant uses its frozen prompt" in result.output


def _make_corpus_episode(corpus_dir, ep_id):
    import json
    from benchmark.corpus import EpisodeMetadata, write_metadata, write_windows, compute_windows, hash_segments

    segments = [
        {"start": 0.0, "end": 5.0, "text": "This episode is brought to you by BetterHelp"},
        {"start": 5.0, "end": 60.0, "text": "Welcome back everyone"},
    ]
    ep_dir = corpus_dir / ep_id
    ep_dir.mkdir(parents=True)
    (ep_dir / "segments.json").write_text(json.dumps(segments))
    write_metadata(ep_dir, EpisodeMetadata(
        ep_id=ep_id, podcast_slug="test-show", podcast_name="Test Show",
        episode_id="abc123", title="Test Episode",
        duration=float(segments[-1]["end"]), segments_hash=hash_segments(segments),
    ))
    (ep_dir / "truth.txt").write_text("start: 0\nend: 5\ntext: BetterHelp\n")
    write_windows(ep_dir, compute_windows(segments))
    return ep_dir
