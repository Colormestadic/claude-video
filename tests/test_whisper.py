"""Whisper auto-chunking: plan, split, and timestamp stitching."""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest

import whisper


MB = 1024 * 1024


class TestPlanChunks:
    def test_under_limit_is_single_chunk(self):
        plan = whisper.plan_chunks(total_seconds=600.0, total_bytes=5 * MB, max_bytes=24 * MB)
        assert plan == [(0.0, 600.0)]

    def test_at_limit_is_single_chunk(self):
        plan = whisper.plan_chunks(total_seconds=600.0, total_bytes=24 * MB, max_bytes=24 * MB)
        assert plan == [(0.0, 600.0)]

    def test_over_limit_splits_into_enough_chunks(self):
        # 71 MB against a 24 MB cap → ceil(71/24) = 3 chunks.
        plan = whisper.plan_chunks(total_seconds=3600.0, total_bytes=71 * MB, max_bytes=24 * MB)
        assert len(plan) == 3

    def test_chunks_are_contiguous_and_cover_full_duration(self):
        total = 3600.0
        plan = whisper.plan_chunks(total_seconds=total, total_bytes=71 * MB, max_bytes=24 * MB)
        # Offsets start at 0 and each picks up where the previous ended.
        assert plan[0][0] == 0.0
        for (off, dur), (next_off, _) in zip(plan, plan[1:]):
            assert math.isclose(off + dur, next_off)
        last_off, last_dur = plan[-1]
        assert math.isclose(last_off + last_dur, total)

    def test_each_chunk_estimated_under_limit(self):
        total_seconds, total_bytes, cap = 3600.0, 71 * MB, 24 * MB
        plan = whisper.plan_chunks(total_seconds, total_bytes, cap)
        bytes_per_second = total_bytes / total_seconds
        for _off, dur in plan:
            assert dur * bytes_per_second <= cap

    def test_zero_duration_is_single_chunk(self):
        plan = whisper.plan_chunks(total_seconds=0.0, total_bytes=0, max_bytes=24 * MB)
        assert plan == [(0.0, 0.0)]


class TestShiftSegments:
    def test_adds_offset_to_start_and_end(self):
        segs = [{"start": 0.0, "end": 2.5, "text": "hi"}, {"start": 2.5, "end": 4.0, "text": "there"}]
        shifted = whisper.shift_segments(segs, 1800.0)
        assert shifted == [
            {"start": 1800.0, "end": 1802.5, "text": "hi"},
            {"start": 1802.5, "end": 1804.0, "text": "there"},
        ]

    def test_zero_offset_is_identity(self):
        segs = [{"start": 1.0, "end": 2.0, "text": "x"}]
        assert whisper.shift_segments(segs, 0.0) == segs

    def test_does_not_mutate_input(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "x"}]
        whisper.shift_segments(segs, 10.0)
        assert segs[0]["start"] == 0.0


def _make_mp3(path: Path, seconds: float) -> None:
    """Synthesize a mono 16k 64k mp3 of a sine tone — mirrors extract_audio's format."""
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-t", str(seconds), "-i", "sine=frequency=440:sample_rate=16000",
            "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", "-b:a", "64k",
            str(path),
        ],
        check=True,
    )


class TestSplitAudio:
    def test_creates_one_file_per_plan_entry(self, tmp_path: Path):
        full = tmp_path / "audio.mp3"
        _make_mp3(full, 6.0)
        plan = [(0.0, 3.0), (3.0, 3.0)]

        chunks = whisper.split_audio(full, tmp_path, plan)

        assert len(chunks) == 2
        for chunk_path, _offset in chunks:
            assert chunk_path.exists() and chunk_path.stat().st_size > 0

    def test_returns_plan_offsets(self, tmp_path: Path):
        full = tmp_path / "audio.mp3"
        _make_mp3(full, 6.0)
        plan = [(0.0, 3.0), (3.0, 3.0)]

        chunks = whisper.split_audio(full, tmp_path, plan)

        assert [offset for _path, offset in chunks] == [0.0, 3.0]

    def test_chunks_are_smaller_than_full(self, tmp_path: Path):
        full = tmp_path / "audio.mp3"
        _make_mp3(full, 6.0)
        plan = [(0.0, 3.0), (3.0, 3.0)]

        chunks = whisper.split_audio(full, tmp_path, plan)

        full_size = full.stat().st_size
        for chunk_path, _offset in chunks:
            assert chunk_path.stat().st_size < full_size


class TestAudioDuration:
    def test_reads_duration_of_synthesized_clip(self, tmp_path: Path):
        audio = tmp_path / "audio.mp3"
        _make_mp3(audio, 5.0)
        assert whisper.audio_duration(audio) == pytest.approx(5.0, abs=0.5)


class TestTranscribeChunks:
    def test_shifts_and_concatenates_each_chunk(self):
        chunks = [(Path("a.mp3"), 0.0), (Path("b.mp3"), 100.0)]

        def fake_transcribe(path: Path) -> list[dict]:
            return [{"start": 0.0, "end": 2.0, "text": path.stem}]

        out = whisper.transcribe_chunks(chunks, fake_transcribe)

        assert out == [
            {"start": 0.0, "end": 2.0, "text": "a"},
            {"start": 100.0, "end": 102.0, "text": "b"},
        ]

    def test_keeps_successful_chunks_when_one_fails(self):
        chunks = [(Path("a.mp3"), 0.0), (Path("b.mp3"), 100.0)]

        def flaky(path: Path) -> list[dict]:
            if path.stem == "b":
                raise SystemExit("chunk b failed")
            return [{"start": 1.0, "end": 2.0, "text": "a"}]

        out = whisper.transcribe_chunks(chunks, flaky)

        assert out == [{"start": 1.0, "end": 2.0, "text": "a"}]

    def test_raises_when_every_chunk_fails(self):
        chunks = [(Path("a.mp3"), 0.0), (Path("b.mp3"), 100.0)]

        def always_fail(path: Path) -> list[dict]:
            raise SystemExit("boom")

        with pytest.raises(SystemExit):
            whisper.transcribe_chunks(chunks, always_fail)


class TestLocalBackend:
    """The CMS fork's keyless offline backend.

    Every test pins WATCH_LOCAL_WHISPER_BIN and clears the API-key env vars, so
    results never depend on what the developer's machine happens to have.
    """

    @staticmethod
    def _fake_whisper(tmp_path: Path, payload: str = "", exit_code: int = 0) -> Path:
        """A stand-in CLI that writes <stem>.json into --output_dir, like the real one."""
        script = tmp_path / "fake-whisper"
        script.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "from pathlib import Path\n"
            f"if {exit_code} != 0:\n"
            "    print('boom', file=sys.stderr)\n"
            f"    raise SystemExit({exit_code})\n"
            "args = sys.argv[1:]\n"
            "audio = Path(args[0])\n"
            "out = Path(args[args.index('--output_dir') + 1])\n"
            "out.mkdir(parents=True, exist_ok=True)\n"
            f"payload = {payload!r}\n"
            "(out / (audio.stem + '.json')).write_text(payload, encoding='utf-8')\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        return script

    @pytest.fixture(autouse=True)
    def _no_api_keys(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # load_api_key also reads ~/.config/watch/.env; point HOME at nothing.
        monkeypatch.setattr(whisper, "load_api_key", lambda preferred=None: (None, None))

    def test_resolve_falls_back_to_local_when_no_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WATCH_LOCAL_WHISPER_BIN", str(self._fake_whisper(tmp_path)))
        assert whisper.resolve_backend() == ("local", None)

    def test_resolve_returns_nothing_when_local_absent(self, monkeypatch):
        monkeypatch.setenv("WATCH_LOCAL_WHISPER_BIN", "/nonexistent/whisper")
        assert whisper.resolve_backend() == (None, None)

    def test_forced_local_needs_the_binary(self, monkeypatch):
        monkeypatch.setenv("WATCH_LOCAL_WHISPER_BIN", "/nonexistent/whisper")
        assert whisper.resolve_backend("local") == (None, None)

    def test_api_key_wins_over_local_in_auto_mode(self, tmp_path, monkeypatch):
        """Local is the fallback, not an override — large-v3 beats a small local model."""
        monkeypatch.setenv("WATCH_LOCAL_WHISPER_BIN", str(self._fake_whisper(tmp_path)))
        monkeypatch.setattr(whisper, "load_api_key", lambda preferred=None: ("groq", "sk-test"))
        assert whisper.resolve_backend() == ("groq", "sk-test")

    def test_parses_real_whisper_cli_json_shape(self, tmp_path, monkeypatch):
        """The CLI's JSON carries the same {segments:[{start,end,text}]} the APIs return."""
        payload = json.dumps({
            "text": " full thing",
            "language": "en",
            "segments": [
                {"id": 0, "seek": 0, "start": 0.0, "end": 8.76, "text": " first line",
                 "tokens": [1, 2], "temperature": 0.0, "avg_logprob": -0.3,
                 "compression_ratio": 1.4, "no_speech_prob": 0.01},
                {"id": 1, "seek": 0, "start": 8.76, "end": 12.5, "text": " second line",
                 "tokens": [3], "temperature": 0.0, "avg_logprob": -0.2,
                 "compression_ratio": 1.2, "no_speech_prob": 0.02},
            ],
        })
        monkeypatch.setenv("WATCH_LOCAL_WHISPER_BIN", str(self._fake_whisper(tmp_path, payload)))
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"\x00")

        segments = whisper._transcribe_local(audio)
        assert segments == [
            {"start": 0.0, "end": 8.76, "text": "first line"},
            {"start": 8.76, "end": 12.5, "text": "second line"},
        ]

    def test_dispatches_through_transcribe_file_with_no_key(self, tmp_path, monkeypatch):
        payload = json.dumps({"text": "hi", "segments": [
            {"start": 0.0, "end": 1.0, "text": " hi"}]})
        monkeypatch.setenv("WATCH_LOCAL_WHISPER_BIN", str(self._fake_whisper(tmp_path, payload)))
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"\x00")

        assert whisper._transcribe_file("local", None, audio) == [
            {"start": 0.0, "end": 1.0, "text": "hi"}
        ]

    def test_cli_failure_surfaces_as_systemexit(self, tmp_path, monkeypatch):
        monkeypatch.setenv(
            "WATCH_LOCAL_WHISPER_BIN", str(self._fake_whisper(tmp_path, exit_code=2))
        )
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"\x00")
        with pytest.raises(SystemExit, match="local whisper failed"):
            whisper._transcribe_local(audio)

    def test_missing_binary_is_a_clear_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WATCH_LOCAL_WHISPER_BIN", "/nonexistent/whisper")
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"\x00")
        with pytest.raises(SystemExit, match="Local whisper is not installed"):
            whisper._transcribe_local(audio)

    def test_model_is_configurable(self, monkeypatch):
        monkeypatch.delenv("WATCH_LOCAL_MODEL", raising=False)
        assert whisper.local_model() == whisper.LOCAL_MODEL_DEFAULT
        monkeypatch.setenv("WATCH_LOCAL_MODEL", "medium.en")
        assert whisper.local_model() == "medium.en"
