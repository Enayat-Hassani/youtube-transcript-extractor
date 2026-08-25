from __future__ import annotations

import sys
from pathlib import Path

import pytest

import ytx_core.ocr as ocr_module
from ytx_core.ocr import OcrEngine, _plausible, available_engine, ocr_lines


def _tesseract_tsv(rows: list[tuple[int, int, int, float, str]]) -> str:
    """Build a tesseract TSV payload: (block, par, line, conf, text)."""
    header = "level\tpage\tblock\tpar\tline\tword\tleft\ttop\twidth\theight\tconf\ttext"
    body = [
        f"5\t1\t{block}\t{par}\t{line}\t1\t0\t0\t0\t0\t{conf}\t{text}"
        for block, par, line, conf, text in rows
    ]
    return "\n".join([header, *body])


def _vision_tsv(rows: list[tuple[float, float, float, str]]) -> str:
    """Build a Vision payload: (x, y, height, text)."""
    return "\n".join(
        f"{x:.5f}\t{y:.5f}\t0.10000\t{height:.5f}\t0.900\t{text}" for x, y, height, text in rows
    )


@pytest.fixture(autouse=True)
def clear_caches():
    ocr_module.available_engine.cache_clear()
    ocr_module._vision_binary.cache_clear()
    yield
    ocr_module.available_engine.cache_clear()
    ocr_module._vision_binary.cache_clear()


class TestPlausible:
    @pytest.mark.parametrize(
        "token", ["$5,537.73", "1.85", "232%", "2007.05.01-2019.12.31", "342"]
    )
    def test_keeps_numbers_and_prices(self, token: str) -> None:
        assert _plausible(token)

    @pytest.mark.parametrize("token", ["Drawdown", "Tradestation", "Sharpe", "OOS", "AAPL"])
    def test_keeps_words_and_acronyms(self, token: str) -> None:
        assert _plausible(token)

    @pytest.mark.parametrize("token", ["»", "|", "~", "x", "‘", "—", "\\"])
    def test_drops_debris(self, token: str) -> None:
        assert not _plausible(token)


class TestEngineSelection:
    def test_prefers_apple_vision(self, monkeypatch) -> None:
        monkeypatch.setattr(ocr_module, "_vision_binary", lambda: Path("/tmp/vision"))
        assert available_engine().name == "apple-vision"

    def test_falls_back_to_tesseract(self, monkeypatch) -> None:
        monkeypatch.setattr(ocr_module, "_vision_binary", lambda: None)
        monkeypatch.setattr(ocr_module.shutil, "which", lambda name: "/usr/bin/tesseract")
        assert available_engine().name == "tesseract"

    def test_reports_when_nothing_is_available(self, monkeypatch) -> None:
        monkeypatch.setattr(ocr_module, "_vision_binary", lambda: None)
        monkeypatch.setattr(ocr_module.shutil, "which", lambda name: None)
        engine = available_engine()
        assert not engine
        assert "tesseract" in (engine.unavailable or "")

    def test_vision_is_skipped_off_darwin(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "platform", "linux")
        assert ocr_module._vision_binary() is None

    def test_vision_is_skipped_without_swiftc(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(ocr_module.shutil, "which", lambda name: None)
        assert ocr_module._vision_binary() is None


class TestVisionLines:
    def _lines(self, monkeypatch, payload: str) -> list[str]:
        monkeypatch.setattr(ocr_module, "_vision_binary", lambda: Path("/tmp/vision"))
        monkeypatch.setattr(ocr_module, "_run", lambda command: payload)
        return ocr_lines(Path("frame.jpg"))

    def test_merges_regions_sharing_a_baseline(self, monkeypatch) -> None:
        payload = _vision_tsv(
            [
                (0.10, 0.800, 0.02, "Engine"),
                (0.40, 0.801, 0.02, "Tradestation"),
                (0.10, 0.700, 0.02, "Symbol"),
                (0.40, 0.700, 0.02, "AAPL.D"),
            ]
        )
        assert self._lines(monkeypatch, payload) == ["Engine Tradestation", "Symbol AAPL.D"]

    def test_orders_lines_top_to_bottom(self, monkeypatch) -> None:
        payload = _vision_tsv(
            [(0.1, 0.20, 0.02, "bottom row"), (0.1, 0.90, 0.02, "top row here")]
        )
        assert self._lines(monkeypatch, payload) == ["top row here", "bottom row"]

    def test_orders_cells_left_to_right(self, monkeypatch) -> None:
        payload = _vision_tsv(
            [(0.7, 0.5, 0.02, "third"), (0.1, 0.5, 0.02, "first"), (0.4, 0.5, 0.02, "second")]
        )
        assert self._lines(monkeypatch, payload) == ["first second third"]

    def test_drops_very_short_lines(self, monkeypatch) -> None:
        assert self._lines(monkeypatch, _vision_tsv([(0.1, 0.5, 0.02, "ok")])) == []

    def test_tolerates_a_dead_helper(self, monkeypatch) -> None:
        monkeypatch.setattr(ocr_module, "_vision_binary", lambda: Path("/tmp/vision"))
        monkeypatch.setattr(ocr_module, "_run", lambda command: None)
        assert ocr_lines(Path("frame.jpg")) == []

    def test_ignores_malformed_rows(self, monkeypatch) -> None:
        payload = "not\tenough\tfields\n" + _vision_tsv([(0.1, 0.5, 0.02, "good line here")])
        assert self._lines(monkeypatch, payload) == ["good line here"]


class TestTesseractLines:
    def _lines(self, monkeypatch, payload: str | None) -> list[str]:
        monkeypatch.setattr(ocr_module, "_vision_binary", lambda: None)
        monkeypatch.setattr(ocr_module.shutil, "which", lambda name: "/usr/bin/tesseract")
        monkeypatch.setattr(ocr_module, "_run", lambda command: payload)
        return ocr_lines(Path("frame.jpg"))

    def test_groups_words_into_lines(self, monkeypatch) -> None:
        payload = _tesseract_tsv(
            [
                (1, 1, 1, 95.0, "Sharpe"),
                (1, 1, 1, 92.0, "Ratio"),
                (1, 1, 2, 90.0, "Net"),
                (1, 1, 2, 90.0, "profit"),
                (1, 1, 2, 88.0, "$5,537.73"),
            ]
        )
        assert self._lines(monkeypatch, payload) == ["Sharpe Ratio", "Net profit $5,537.73"]

    def test_drops_low_confidence_words(self, monkeypatch) -> None:
        payload = _tesseract_tsv(
            [(1, 1, 1, 95.0, "Profit"), (1, 1, 1, 12.0, "zzz"), (1, 1, 1, 91.0, "factor")]
        )
        assert self._lines(monkeypatch, payload) == ["Profit factor"]

    def test_drops_debris_tokens(self, monkeypatch) -> None:
        payload = _tesseract_tsv(
            [(1, 1, 1, 95.0, "Profit"), (1, 1, 1, 95.0, "»"), (1, 1, 1, 95.0, "factor")]
        )
        assert self._lines(monkeypatch, payload) == ["Profit factor"]

    def test_uses_sparse_page_segmentation(self, monkeypatch) -> None:
        seen: list[list[str]] = []
        monkeypatch.setattr(ocr_module, "_vision_binary", lambda: None)
        monkeypatch.setattr(ocr_module.shutil, "which", lambda name: "/usr/bin/tesseract")
        monkeypatch.setattr(ocr_module, "_run", lambda command: seen.append(command) or "")
        ocr_lines(Path("frame.jpg"))
        assert "--psm" in seen[0]
        assert seen[0][seen[0].index("--psm") + 1] == "11"

    def test_tolerates_a_failed_run(self, monkeypatch) -> None:
        assert self._lines(monkeypatch, None) == []


def test_no_engine_means_no_lines(monkeypatch) -> None:
    monkeypatch.setattr(ocr_module, "_vision_binary", lambda: None)
    monkeypatch.setattr(ocr_module.shutil, "which", lambda name: None)
    assert ocr_lines(Path("frame.jpg")) == []


def test_engine_dataclass_truthiness() -> None:
    assert OcrEngine("apple-vision")
    assert not OcrEngine("none", unavailable="missing")
