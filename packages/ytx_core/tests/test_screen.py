from __future__ import annotations

from pathlib import Path

import pytest

import ytx_core.screen as screen_module
from ytx_core.ocr import OcrEngine
from ytx_core.screen import (
    ScreenCapture,
    _drop_repeats,
    _strip_boilerplate,
    extract_screen_text,
    tools_available,
)


class TestSegmentsAndMoments:
    def test_cuts_become_segments_spanning_the_video(self) -> None:
        assert screen_module._segments([30.0, 90.0], 120.0) == [
            (0.0, 30.0),
            (30.0, 90.0),
            (90.0, 120.0),
        ]

    def test_ignores_cuts_outside_the_video(self) -> None:
        assert screen_module._segments([-5.0, 60.0, 999.0], 120.0) == [
            (0.0, 60.0),
            (60.0, 120.0),
        ]

    def test_no_cuts_means_one_segment(self) -> None:
        assert screen_module._segments([], 120.0) == [(0.0, 120.0)]

    def test_samples_the_middle_of_a_segment_not_the_cut(self) -> None:
        # A frame taken at a cut lands mid-transition.
        assert screen_module._moments([(0.0, 60.0)], 120.0, 10) == [30.0]

    def test_subdivides_a_segment_longer_than_the_interval(self) -> None:
        moments = screen_module._moments([(0.0, 300.0)], 100.0, 10)
        assert moments == [50.0, 150.0, 250.0]

    def test_short_segments_each_get_a_sample(self) -> None:
        segments = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
        assert screen_module._moments(segments, 120.0, 10) == [5.0, 15.0, 25.0]

    def test_over_budget_prefers_the_longest_segments(self) -> None:
        # Rapid cuts (conversation) plus two long holds (a screen).
        segments = [(float(i), float(i + 1)) for i in range(50)]
        segments += [(100.0, 160.0), (200.0, 280.0)]
        moments = screen_module._moments(segments, 120.0, 2)
        assert moments == [130.0, 240.0]

    def test_under_budget_keeps_everything_in_time_order(self) -> None:
        segments = [(0.0, 10.0), (10.0, 30.0), (30.0, 36.0)]
        assert screen_module._moments(segments, 120.0, 10) == [5.0, 20.0, 33.0]

    def test_subdivisions_inherit_their_segment_length(self) -> None:
        # A long segment's samples must outrank short segments' samples.
        segments = [(0.0, 300.0)] + [(float(300 + i), float(301 + i)) for i in range(50)]
        moments = screen_module._moments(segments, 100.0, 3)
        assert moments == [50.0, 150.0, 250.0]

    def test_no_segments_means_no_moments(self) -> None:
        assert screen_module._moments([], 120.0, 10) == []


class TestFiltering:
    def test_states_a_recurring_line_once(self) -> None:
        words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
        raw = [(float(i), ["Builder Progress Full settings", f"only {w} here"])
               for i, w in enumerate(words)]
        cleaned = _strip_boilerplate(raw)
        chrome = [line for _t, lines in cleaned for line in lines if "Builder" in line]
        assert len(chrome) == 1
        assert all(any("only" in line for line in lines) for _t, lines in cleaned)

    def test_matches_chrome_the_ocr_reads_differently_each_time(self) -> None:
        # The same title bar, misread slightly differently on every frame.
        readings = [
            "ChartFonatics StrategyQuant X Pro Bunld 142 trial",
            "ChartFanatics StrategyQuant X Pro Build 142 trial",
            "ChartFonotics StrategyQuant X Pro Bunld T42 trial",
            "ChartFanotics StrategyQuant X Pro Build 142 tral",
            "ChartFonatics StrategyQuant X Pro Bunld 142 trial",
        ]
        raw = [(float(i), [reading, f"body {w}"])
               for i, (reading, w) in enumerate(zip(readings, "abcde", strict=True))]
        cleaned = _strip_boilerplate(raw)
        titles = [
            line
            for _t, lines in cleaned
            for line in lines
            if "trial" in line or "tral" in line
        ]
        assert len(titles) == 1

    def test_matches_chrome_whose_digits_change(self) -> None:
        # A title bar with a running clock is the same line every frame.
        raw = [
            (float(i), [f"StrategyQuant X Pro Build 142 trial valid until 0{i}", "body text"])
            for i in range(6)
        ]
        cleaned = _strip_boilerplate(raw)
        titles = [line for _t, lines in cleaned for line in lines if "StrategyQuant" in line]
        assert len(titles) == 1

    def test_never_collapses_data_rows_that_share_words(self) -> None:
        # Same words, different numbers: distinct backtest results, all kept.
        raw = [
            (0.0, ["Strategy 1.8.186 net 5537.73 sharpe 1.02 drawdown 480.76"]),
            (10.0, ["Strategy 1.8.187 net 4275.44 sharpe 0.84 drawdown 697.68"]),
            (20.0, ["Strategy 1.8.188 net 6955.99 sharpe 1.09 drawdown 817.88"]),
            (30.0, ["Strategy 1.8.189 net 4972.10 sharpe 0.79 drawdown 818.19"]),
        ]
        cleaned = _strip_boilerplate(raw)
        assert sum(len(lines) for _t, lines in cleaned) == 4

    def test_keeps_everything_when_there_are_few_captures(self) -> None:
        raw = [(0.0, ["a line here"]), (1.0, ["a line here"])]
        assert _strip_boilerplate(raw) == raw

    def test_collapses_an_unchanged_screen(self) -> None:
        raw = [
            (0.0, ["Profit factor 1.85 Sharpe 1.02"]),
            (10.0, ["Profit factor 1.85 Sharpe 1.02"]),
            (20.0, ["Completely different content appears now"]),
        ]
        kept = _drop_repeats(raw)
        assert [time for time, _ in kept] == [0.0, 20.0]


class TestExtractScreenText:
    def test_reports_a_missing_tool(self, monkeypatch) -> None:
        monkeypatch.setattr(screen_module.shutil, "which", lambda name: None)
        result = extract_screen_text("abc", 600.0)
        assert result.captures == []
        assert "ffmpeg" in result.status
        assert not result

    def test_reports_a_missing_ocr_engine(self, monkeypatch) -> None:
        monkeypatch.setattr(screen_module.shutil, "which", lambda name: "/bin/ffmpeg")
        monkeypatch.setattr(
            screen_module, "available_engine", lambda: OcrEngine("none", unavailable="no OCR")
        )
        assert extract_screen_text("abc", 600.0).status == "no OCR"

    def test_reports_an_unresolvable_stream(self, monkeypatch) -> None:
        monkeypatch.setattr(screen_module.shutil, "which", lambda name: "/bin/x")
        monkeypatch.setattr(screen_module, "_stream_url", lambda video_id: None)
        assert "stream" in extract_screen_text("abc", 600.0).status

    def test_reports_an_unknown_duration(self, monkeypatch) -> None:
        monkeypatch.setattr(screen_module.shutil, "which", lambda name: "/bin/x")
        assert "duration" in extract_screen_text("abc", 0.0).status

    def test_reports_a_video_with_no_screen_text(self, monkeypatch) -> None:
        monkeypatch.setattr(screen_module.shutil, "which", lambda name: "/bin/x")
        monkeypatch.setattr(
            screen_module, "_source", lambda vid, dur, workdir: ("http://x", False)
        )
        monkeypatch.setattr(screen_module, "_scene_cuts", lambda url, workdir: None)
        monkeypatch.setattr(
            screen_module, "_grab", lambda url, at, path: path.write_bytes(b"x") or True
        )
        monkeypatch.setattr(screen_module, "ocr_lines", lambda path: [])
        result = extract_screen_text("abc", 600.0)
        assert result.captures == []
        assert "no on-screen text" in result.status

    def test_extracts_and_deletes_every_frame(self, monkeypatch) -> None:
        seen: list[Path] = []

        def grab(url, at, path):
            seen.append(path)
            path.write_bytes(b"jpeg")
            return True

        monkeypatch.setattr(screen_module.shutil, "which", lambda name: "/bin/x")
        monkeypatch.setattr(
            screen_module, "_source", lambda vid, dur, workdir: ("http://x", False)
        )
        monkeypatch.setattr(screen_module, "_scene_cuts", lambda url, workdir: None)
        monkeypatch.setattr(screen_module, "_grab", grab)
        monkeypatch.setattr(
            screen_module,
            "ocr_lines",
            lambda path: [
                "Engine Tradestation Symbol AAPL.D timeframe D1 range 2007.05.01-2019.12.31",
                "Net profit $5,537.73 Sharpe Ratio 1.02 Profit factor 1.85 trades 342",
                "Drawdown $480.76 Annual return 1.85% Stability 0.57 Win Loss 1.63",
                f"Builder Progress Full settings Results task running at {path.stem}",
            ],
        )
        result = extract_screen_text("abc", 1200.0, interval=120.0, max_frames=6)

        assert result.captures
        assert any("Sharpe" in capture.text for capture in result.captures)
        # Nothing survives on disk, and the whole scratch directory is gone.
        assert seen and not any(path.exists() for path in seen)
        assert not seen[0].parent.exists()

    def test_keep_frames_saves_copies_outside_the_scratch_dir(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(screen_module.shutil, "which", lambda name: "/bin/x")
        monkeypatch.setattr(
            screen_module, "_source", lambda vid, dur, workdir: ("http://x", False)
        )
        monkeypatch.setattr(screen_module, "_scene_cuts", lambda url, workdir: None)
        monkeypatch.setattr(
            screen_module, "_grab", lambda url, at, path: path.write_bytes(b"jpeg") or True
        )
        monkeypatch.setattr(
            screen_module,
            "ocr_lines",
            lambda path: [
                "Engine Tradestation Symbol AAPL.D timeframe D1 range 2007.05.01-2019.12.31",
                "Net profit $5,537.73 Sharpe Ratio 1.02 Profit factor 1.85 trades 342",
                "Drawdown $480.76 Annual return 1.85% Stability 0.57 Win Loss 1.63",
                f"Builder Progress Full settings Results at {path.stem}",
            ],
        )
        target = tmp_path / "frames"
        extract_screen_text("abc", 600.0, interval=200.0, max_frames=3, keep_frames=target)
        assert target.is_dir()
        assert list(target.glob("abc-*.jpg"))

    def test_respects_the_frame_cap(self, monkeypatch) -> None:
        calls: list[float] = []

        def grab(url, at, path):
            calls.append(at)
            path.write_bytes(b"jpeg")
            return True

        monkeypatch.setattr(screen_module.shutil, "which", lambda name: "/bin/x")
        monkeypatch.setattr(
            screen_module, "_source", lambda vid, dur, workdir: ("http://x", False)
        )
        monkeypatch.setattr(screen_module, "_scene_cuts", lambda url, workdir: None)
        monkeypatch.setattr(screen_module, "_grab", grab)
        monkeypatch.setattr(
            screen_module,
            "ocr_lines",
            lambda path: [
                "Engine Tradestation Symbol AAPL.D timeframe D1 range 2007.05.01-2019.12.31",
                "Net profit $5,537.73 Sharpe Ratio 1.02 Profit factor 1.85 trades 342",
                f"Drawdown $480.76 Annual return 1.85% Stability at {path.stem}",
            ],
        )
        extract_screen_text("abc", 100_000.0, interval=10.0, max_frames=5)
        assert len(calls) <= 5

    def test_a_failing_frame_is_skipped_not_fatal(self, monkeypatch) -> None:
        monkeypatch.setattr(screen_module.shutil, "which", lambda name: "/bin/x")
        monkeypatch.setattr(
            screen_module, "_source", lambda vid, dur, workdir: ("http://x", False)
        )
        monkeypatch.setattr(screen_module, "_scene_cuts", lambda url, workdir: None)
        monkeypatch.setattr(screen_module, "_grab", lambda url, at, path: False)
        result = extract_screen_text("abc", 600.0)
        assert result.captures == []
        assert "no frames" in result.status


def test_screen_capture_is_hashable_and_immutable() -> None:
    capture = ScreenCapture(time=1.0, text="x")
    with pytest.raises(AttributeError):
        capture.text = "y"  # type: ignore[misc]


def test_tools_available_names_the_missing_one(monkeypatch) -> None:
    monkeypatch.setattr(screen_module.shutil, "which", lambda name: None)
    assert tools_available() == "ffmpeg"


class TestSourceResolution:
    def test_prefers_a_local_download(self, monkeypatch, tmp_path) -> None:
        def fake_download(vid, target):
            target.write_bytes(b"v")
            return target

        monkeypatch.setattr(screen_module, "_download", fake_download)
        monkeypatch.setattr(
            screen_module, "_stream_url", lambda vid: pytest.fail("should not stream")
        )
        url, local = screen_module._source("abc", 600.0, tmp_path)
        assert local is True
        assert url.endswith("video.mp4")

    def test_falls_back_to_streaming_when_the_download_fails(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(screen_module, "_download", lambda vid, target: None)
        monkeypatch.setattr(screen_module, "_stream_url", lambda vid: "http://stream")
        assert screen_module._source("abc", 600.0, tmp_path) == ("http://stream", False)

    def test_streams_rather_than_downloading_a_very_long_video(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(
            screen_module, "_download", lambda vid, target: pytest.fail("should not download")
        )
        monkeypatch.setattr(screen_module, "_stream_url", lambda vid: "http://stream")
        hours = screen_module._MAX_DOWNLOAD_SECONDS + 1
        assert screen_module._source("abc", hours, tmp_path) == ("http://stream", False)
