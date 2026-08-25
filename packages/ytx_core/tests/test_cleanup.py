from __future__ import annotations

import json
import urllib.error
from io import BytesIO

import pytest

import ytx_core.cleanup as cleanup_pkg
from conftest import make_document, make_segment
from ytx_core.cleanup import CleanupOptions, clean
from ytx_core.cleanup.lexicon import (
    GENERAL_PACK,
    Lexicon,
    PackError,
    available_packs,
    build_lexicon,
    detect_packs,
    load_pack_file,
)
from ytx_core.cleanup.sponsorblock import fetch_ranges
from ytx_core.cleanup.sponsors import (
    SponsorRange,
    detect_sponsor_ranges,
    remove_ranges,
    strip_description_links,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """No test may reach SponsorBlock; opt in explicitly where needed."""
    monkeypatch.setattr(cleanup_pkg, "fetch_ranges", lambda *a, **k: [])


TRADING = build_lexicon(["trading"])
TECH = build_lexicon(["tech"])


class TestPacks:
    def test_bundled_packs_load(self) -> None:
        packs = available_packs()
        assert {"general", "trading", "tech"} <= set(packs)
        assert all(pack.rules for pack in packs.values())

    def test_general_is_always_applied_and_last(self) -> None:
        assert build_lexicon([]).names == (GENERAL_PACK,)
        assert build_lexicon(["trading"]).names == ("trading", GENERAL_PACK)

    def test_unknown_pack_is_rejected(self) -> None:
        with pytest.raises(PackError, match="unknown lexicon pack"):
            build_lexicon(["nonsense"])

    def test_custom_pack_outranks_bundled(self, tmp_path) -> None:
        path = tmp_path / "mine.toml"
        path.write_text(
            'name = "mine"\nrules = [[\'\\bback\\s+test\\b\', "BESPOKE"]]\n',
            encoding="utf-8",
        )
        lexicon = build_lexicon(["trading"], extra=[load_pack_file(path)])
        assert lexicon.names == ("mine", "trading", GENERAL_PACK)
        assert lexicon.fix("you back test it") == "you BESPOKE it"

    def test_missing_custom_pack_errors(self, tmp_path) -> None:
        with pytest.raises(PackError, match="not found"):
            load_pack_file(tmp_path / "absent.toml")

    def test_malformed_pack_errors(self, tmp_path) -> None:
        path = tmp_path / "bad.toml"
        path.write_text('name = "bad"\nrules = [["only-one-element"]]\n', encoding="utf-8")
        with pytest.raises(PackError, match="pattern, replacement"):
            load_pack_file(path)

    def test_invalid_regex_is_rejected_at_load(self, tmp_path) -> None:
        path = tmp_path / "bad.toml"
        path.write_text('name = "bad"\nrules = [["(unclosed", "x"]]\n', encoding="utf-8")
        with pytest.raises(PackError, match="invalid pattern"):
            load_pack_file(path)

    def test_empty_lexicon_is_a_no_op(self) -> None:
        assert Lexicon([]).fix("back test") == "back test"


class TestDetectPacks:
    def test_picks_the_matching_domain(self) -> None:
        assert detect_packs(
            "algo trading, backtest, drawdown, futures, position size"
        ) == ("trading",)
        assert detect_packs(
            "we deploy the api with docker, kubernetes, python and typescript"
        ) == ("tech",)

    def test_returns_nothing_for_an_unrelated_video(self) -> None:
        assert detect_packs("baking sourdough bread at home") == ()

    def test_drops_a_weak_secondary_pack(self) -> None:
        text = (
            "trading trader backtest drawdown futures algo forex scalping "
            "candlestick — mentions code and software in passing"
        )
        assert detect_packs(text) == ("trading",)

    def test_keeps_both_when_genuinely_dual_domain(self) -> None:
        text = (
            "algo trading trader backtest drawdown futures forex position size "
            "built in python with an api, docker, kubernetes, a backend and deploy"
        )
        assert set(detect_packs(text)) == {"trading", "tech"}


class TestLexiconRules:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("the sharp ratio is high", "the Sharpe ratio is high"),
            ("a good aggro trader", "a good algo trader"),
            ("max draw down of 10%", "max drawdown of 10%"),
            ("you back test it", "you backtest it"),
            ("when back testing", "when backtesting"),
            ("use rsi and macd", "use RSI and MACD"),
            ("on a lower time frame", "on a lower timeframe"),
            ("monte carlo analysis", "Monte Carlo analysis"),
            ("out of sample data", "out-of-sample data"),
        ],
    )
    def test_trading_rules(self, raw: str, expected: str) -> None:
        assert TRADING.fix(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("deploy to engine x", "deploy to nginx"),
            ("a java script app", "a JavaScript app"),
            ("a postgres ql database", "a PostgreSQL database"),
            ("push it to git hub", "push it to GitHub"),
            ("call the rest api", "call the REST API"),
            ("we use o auth", "we use OAuth"),
        ],
    )
    def test_tech_rules(self, raw: str, expected: str) -> None:
        assert TECH.fix(raw) == expected

    def test_general_rules_apply_to_every_lexicon(self) -> None:
        for lexicon in (build_lexicon([]), TRADING, TECH):
            assert lexicon.fix("watch it on you tube") == "watch it on YouTube"
            assert lexicon.fix("visit the web site") == "visit the website"

    def test_longest_alternative_wins(self) -> None:
        assert TRADING.fix("i use strategic quant x sqx") == (
            "i use StrategyQuant X StrategyQuant X"
        )

    def test_does_not_double_apply(self) -> None:
        once = TRADING.fix("back testing the drawdown")
        assert once == "backtesting the drawdown"
        assert TRADING.fix(once) == once

    def test_leaves_correct_text_untouched(self) -> None:
        text = "backtesting a drawdown with the Sharpe ratio on a timeframe"
        assert TRADING.fix(text) == text

    def test_preserves_sentence_initial_capital(self) -> None:
        assert TRADING.fix("Back testing matters.") == "Backtesting matters."

    def test_respects_word_boundaries(self) -> None:
        assert TRADING.fix("premature parsing") == "premature parsing"

    def test_apply_counts_changed_segments(self) -> None:
        doc = make_document(
            segments=[
                make_segment(0.0, 1.0, "the sharp ratio"),
                make_segment(1.0, 2.0, "nothing to fix here"),
                make_segment(2.0, 3.0, "back test it"),
            ]
        )
        cleaned, changed = TRADING.apply(doc)
        assert changed == 2
        assert cleaned.segments[0].text == "the Sharpe ratio"
        assert cleaned.segments[2].text == "backtest it"

    def test_apply_returns_original_when_unchanged(self) -> None:
        doc = make_document(segments=[make_segment(0.0, 1.0, "plain speech")])
        cleaned, changed = TRADING.apply(doc)
        assert changed == 0
        assert cleaned is doc

    def test_apply_does_not_mutate_input(self) -> None:
        doc = make_document(segments=[make_segment(0.0, 1.0, "the sharp ratio")])
        TRADING.apply(doc)
        assert doc.segments[0].text == "the sharp ratio"


def _ad_document():
    return make_document(
        segments=[
            make_segment(0.0, 5.0, "Today we talk about risk adjusted returns."),
            make_segment(5.0, 10.0, "You want to size positions by volatility."),
            make_segment(10.0, 15.0, "Before we get into this episode, a quick word."),
            make_segment(15.0, 20.0, "Get 90% off Apex Trader Funding, use code CF."),
            make_segment(20.0, 25.0, "The links are in the description below."),
            make_segment(25.0, 30.0, "Join our discord community and subscribe."),
            make_segment(30.0, 35.0, "Now back to the entry logic and exit logic."),
            make_segment(35.0, 40.0, "We define the stop and the target."),
        ]
    )


def _strip(doc):
    return remove_ranges(doc, detect_sponsor_ranges(doc))


class TestHeuristicDetection:
    def test_removes_a_multi_signal_block(self) -> None:
        cleaned, blocks = _strip(_ad_document())
        texts = [segment.text for segment in cleaned.segments]
        assert len(blocks) == 1
        assert not any("Apex Trader Funding" in text for text in texts)
        assert any("entry logic" in text for text in texts)
        assert any("risk adjusted returns" in text for text in texts)

    def test_reports_source_and_extent(self) -> None:
        _cleaned, blocks = _strip(_ad_document())
        assert blocks[0].sources == ["heuristic"]
        assert blocks[0].segment_count == 4
        assert blocks[0].start == 10.0

    def test_single_category_is_not_enough(self) -> None:
        doc = make_document(
            segments=[
                make_segment(0.0, 5.0, "The strategy is a breakout system."),
                make_segment(5.0, 10.0, "Please subscribe if this helps."),
                make_segment(10.0, 15.0, "We enter on the retest of the level."),
            ]
        )
        cleaned, blocks = _strip(doc)
        assert blocks == []
        assert len(cleaned.segments) == 3

    def test_a_discount_code_qualifies_alone(self) -> None:
        doc = make_document(
            segments=[
                make_segment(0.0, 5.0, "We size positions by volatility."),
                make_segment(5.0, 10.0, "Use the code CF for up to 90% off."),
                make_segment(10.0, 15.0, "Use code CF at checkout, 20% off yearly."),
                make_segment(15.0, 20.0, "Back to the exit logic."),
            ]
        )
        cleaned, blocks = _strip(doc)
        assert len(blocks) == 1
        assert blocks[0].categories == ["promo_code"]
        assert len(cleaned.segments) == 2

    def test_expands_over_segments_naming_the_advertiser(self) -> None:
        doc = make_document(
            segments=[
                make_segment(0.0, 5.0, "We size positions by volatility."),
                make_segment(5.0, 10.0, "The entry rule is a close above the range."),
                make_segment(10.0, 15.0, "We measure the risk on the daily bar."),
                make_segment(15.0, 20.0, "Zellabroker gives you twenty accounts."),
                make_segment(20.0, 25.0, "Use the code CF for 90% off Zellabroker."),
                make_segment(25.0, 30.0, "Use code CF, link in the description below."),
                make_segment(30.0, 35.0, "Zellabroker pays out every eight days."),
                make_segment(35.0, 40.0, "Back to the exit logic."),
                make_segment(40.0, 45.0, "The target is two times the initial risk."),
                make_segment(45.0, 50.0, "That completes the system."),
            ]
        )
        cleaned, blocks = _strip(doc)
        texts = [segment.text for segment in cleaned.segments]
        assert not any("Zellabroker" in text for text in texts)
        assert blocks[0].segment_count == 4
        assert "We size positions by volatility." in texts
        assert "Back to the exit logic." in texts

    def test_expansion_never_crosses_into_unrelated_speech(self) -> None:
        doc = make_document(
            segments=[
                make_segment(0.0, 5.0, "The Sortino ratio measures downside deviation."),
                make_segment(5.0, 10.0, "Use the code CF for 90% off Zellabroker."),
                make_segment(10.0, 15.0, "Use code CF, link in the description below."),
                make_segment(15.0, 20.0, "The Calmar ratio uses maximum drawdown."),
            ]
        )
        cleaned, _blocks = _strip(doc)
        texts = [segment.text for segment in cleaned.segments]
        assert len(cleaned.segments) == 2
        assert "Sortino" in texts[0]
        assert "Calmar" in texts[1]

    def test_rejects_a_block_that_spans_too_long(self) -> None:
        doc = make_document(
            segments=[
                make_segment(0.0, 10.0, "Use code X for a discount."),
                make_segment(600.0, 610.0, "Subscribe and join our discord."),
            ]
        )
        assert detect_sponsor_ranges(doc) == []

    def test_never_strips_more_than_half(self) -> None:
        segments = [
            make_segment(
                float(i), float(i + 1), "Use code CF and subscribe, link in the description below."
            )
            for i in range(20)
        ]
        cleaned, _blocks = _strip(make_document(segments=segments))
        assert len(cleaned.segments) >= 10

    def test_empty_document_is_safe(self) -> None:
        doc = make_document(segments=[make_segment(0.0, 1.0, "hi")])
        empty = doc.model_copy(update={"segments": []})
        assert detect_sponsor_ranges(empty) == []
        assert remove_ranges(empty, []) == (empty, [])


class TestRemoveRanges:
    def test_verified_range_exceeds_the_local_budget(self) -> None:
        # A real 9-minute sponsor read: past the local budget (50 of 200
        # segments) but inside the half-transcript ceiling.
        segments = [
            make_segment(float(i * 10), float(i * 10 + 10), f"line {i}") for i in range(200)
        ]
        doc = make_document(segments=segments)
        verified = SponsorRange(
            start=0.0, end=540.0, sources=["sponsorblock"], categories=["sponsor"]
        )
        cleaned, blocks = remove_ranges(doc, [verified])
        assert blocks and blocks[0].sources == ["sponsorblock"]
        assert blocks[0].segment_count == 54
        assert len(cleaned.segments) == 146

    def test_the_same_range_from_the_heuristic_is_refused(self) -> None:
        segments = [
            make_segment(float(i * 10), float(i * 10 + 10), f"line {i}") for i in range(200)
        ]
        doc = make_document(segments=segments)
        guess = SponsorRange(
            start=0.0, end=540.0, sources=["heuristic"], categories=["promo_code"]
        )
        cleaned, blocks = remove_ranges(doc, [guess])
        assert blocks == []
        assert len(cleaned.segments) == 200

    def test_verified_ranges_still_obey_the_half_transcript_ceiling(self) -> None:
        segments = [make_segment(float(i), float(i + 1), f"line {i}") for i in range(20)]
        doc = make_document(segments=segments)
        verified = SponsorRange(
            start=0.0, end=18.0, sources=["sponsorblock"], categories=["sponsor"]
        )
        cleaned, blocks = remove_ranges(doc, [verified])
        assert blocks == []
        assert len(cleaned.segments) == 20

    def test_overlapping_ranges_merge_and_keep_verified_source(self) -> None:
        doc = make_document(
            segments=[make_segment(float(i * 5), float(i * 5 + 5), f"line {i}") for i in range(12)]
        )
        ranges = [
            SponsorRange(start=10.0, end=25.0, sources=["heuristic"], categories=["promo_code"]),
            SponsorRange(start=20.0, end=35.0, sources=["sponsorblock"], categories=["sponsor"]),
        ]
        _cleaned, blocks = remove_ranges(doc, ranges)
        assert len(blocks) == 1
        assert set(blocks[0].sources) == {"heuristic", "sponsorblock"}
        assert set(blocks[0].categories) == {"promo_code", "sponsor"}

    def test_no_ranges_is_a_no_op(self) -> None:
        doc = _ad_document()
        assert remove_ranges(doc, []) == (doc, [])


class TestSponsorBlockClient:
    def _respond(self, monkeypatch, payload):
        body = BytesIO(json.dumps(payload).encode())
        body.__enter__ = lambda self=body: self  # type: ignore[method-assign]
        body.__exit__ = lambda *a: None  # type: ignore[method-assign]
        monkeypatch.setattr(
            "ytx_core.cleanup.sponsorblock.urllib.request.urlopen", lambda *a, **k: body
        )

    def test_parses_matching_video_only(self, monkeypatch) -> None:
        self._respond(
            monkeypatch,
            [
                {
                    "videoID": "abc",
                    "segments": [
                        {"segment": [10.0, 20.0], "category": "sponsor", "actionType": "skip",
                         "votes": 3}
                    ],
                },
                {
                    "videoID": "other",
                    "segments": [
                        {"segment": [0.0, 5.0], "category": "sponsor", "actionType": "skip",
                         "votes": 1}
                    ],
                },
            ],
        )
        assert fetch_ranges("abc") == [(10.0, 20.0, "sponsor")]

    def test_skips_non_skip_actions_and_downvoted(self, monkeypatch) -> None:
        self._respond(
            monkeypatch,
            [
                {
                    "videoID": "abc",
                    "segments": [
                        {"segment": [0.0, 5.0], "category": "sponsor", "actionType": "mute",
                         "votes": 5},
                        {"segment": [6.0, 9.0], "category": "sponsor", "actionType": "skip",
                         "votes": -2},
                        {"segment": [10.0, 12.0], "category": "selfpromo", "actionType": "skip",
                         "votes": 0},
                    ],
                }
            ],
        )
        assert fetch_ranges("abc") == [(10.0, 12.0, "selfpromo")]

    def test_404_means_not_covered(self, monkeypatch) -> None:
        def raise_404(*a, **k):
            raise urllib.error.HTTPError("u", 404, "nope", {}, None)

        monkeypatch.setattr(
            "ytx_core.cleanup.sponsorblock.urllib.request.urlopen", raise_404
        )
        assert fetch_ranges("abc") == []

    def test_network_failure_returns_none(self, monkeypatch) -> None:
        def boom(*a, **k):
            raise OSError("offline")

        monkeypatch.setattr("ytx_core.cleanup.sponsorblock.urllib.request.urlopen", boom)
        assert fetch_ranges("abc") is None

    def test_full_video_id_is_never_sent(self, monkeypatch) -> None:
        seen: list[str] = []

        def capture(url, *a, **k):
            seen.append(url)
            raise urllib.error.HTTPError("u", 404, "nope", {}, None)

        monkeypatch.setattr("ytx_core.cleanup.sponsorblock.urllib.request.urlopen", capture)
        fetch_ranges("TyHTEtArsS4")
        assert "TyHTEtArsS4" not in seen[0]
        # sha256("TyHTEtArsS4") starts with e8e7
        assert "/e8e7" in seen[0]


class TestStripDescriptionLinks:
    def test_drops_affiliate_link_lines(self) -> None:
        raw = (
            "90% OFF Apex Trader Funding (Code: CF): https://apextraderfunding.com/aff/go\n"
            "Chart Academy: https://www.chartacademy.com/\n"
            "\n"
            "In this episode a trader shares a practical guide to trading.\n"
        )
        cleaned = strip_description_links(raw)
        assert "apextraderfunding" not in cleaned
        assert "practical guide" in cleaned

    def test_drops_a_long_promo_label_wrapped_around_a_link(self) -> None:
        raw = (
            "20% OFF TradeZella (Code: CF10 for 10% OFF Monthly or CF20 for 20% OFF "
            "Yearly): https://refer.tradezella.com/x\n"
            "A sentence of genuine description text that should survive intact."
        )
        cleaned = strip_description_links(raw)
        assert "TradeZella" not in cleaned
        assert "genuine description text" in cleaned

    def test_drops_decorative_separator_lines(self) -> None:
        raw = "Real text.\n━━━━━━\nMore real text."
        cleaned = strip_description_links(raw)
        assert "━" not in cleaned
        assert "Real text." in cleaned and "More real text." in cleaned

    def test_keeps_prose_that_merely_mentions_a_url(self) -> None:
        raw = (
            "This long paragraph explains the whole methodology in detail and "
            "happens to cite example.com as one source among several others."
        )
        assert strip_description_links(raw) == raw

    def test_handles_empty_input(self) -> None:
        assert strip_description_links("") == ""


class TestCleanFacade:
    def test_applies_every_pass_and_reports(self) -> None:
        doc = _ad_document().model_copy(
            update={
                "segments": [
                    *_ad_document().segments,
                    make_segment(40.0, 45.0, "Check the sharp ratio when you back test."),
                ]
            }
        )
        cleaned, description, report = clean(
            doc,
            video_id="abc",
            title="Algo trading: backtest your drawdown on futures",
            description="Sponsor: https://example.com/aff/go\n\nReal prose here.",
        )
        assert report.lexicon_packs == ["trading", "general"]
        assert report.terms_fixed == 1
        assert report.segments_removed == 4
        assert "Sharpe ratio" in cleaned.segments[-1].text
        assert "example.com" not in (description or "")
        assert report.description_trimmed is True
        assert len(report.notes) == 3

    def test_disabled_options_change_nothing(self) -> None:
        doc = _ad_document()
        cleaned, description, report = clean(
            doc, video_id="abc", description="keep https://example.com/aff",
            options=CleanupOptions.disabled(),
        )
        assert cleaned is doc
        assert description == "keep https://example.com/aff"
        assert report.notes == []

    def test_explicit_packs_override_detection(self) -> None:
        doc = make_document(segments=[make_segment(0.0, 1.0, "deploy to engine x")])
        _cleaned, _desc, report = clean(
            doc, video_id="abc", title="a trading video about backtest drawdown futures",
            options=CleanupOptions(packs=("tech",)),
        )
        assert report.lexicon_packs == ["tech", "general"]

    def test_domain_none_leaves_general_only(self) -> None:
        doc = make_document(segments=[make_segment(0.0, 1.0, "the sharp ratio on you tube")])
        cleaned, _desc, report = clean(
            doc, video_id="abc", options=CleanupOptions(packs=())
        )
        assert report.lexicon_packs == ["general"]
        assert cleaned.segments[0].text == "the sharp ratio on YouTube"

    def test_sponsorblock_ranges_are_merged_in(self, monkeypatch) -> None:
        monkeypatch.setattr(
            cleanup_pkg, "fetch_ranges", lambda *a, **k: [(0.0, 12.0, "sponsor")]
        )
        # Pad with content so the merged block fits inside the removal budget.
        doc = _ad_document().model_copy(
            update={
                "segments": [
                    *_ad_document().segments,
                    *(
                        make_segment(float(40 + i * 5), float(45 + i * 5), f"content {i}")
                        for i in range(8)
                    ),
                ]
            }
        )
        cleaned, _desc, report = clean(doc, video_id="abc")
        assert report.sponsorblock == "hit"
        # The verified range overlaps the local one, so the block carries both.
        assert {source for block in report.removed_blocks for source in block.sources} == {
            "sponsorblock",
            "heuristic",
        }
        assert len(cleaned.segments) < len(doc.segments)

    def test_sponsorblock_outage_falls_back_silently(self, monkeypatch) -> None:
        monkeypatch.setattr(cleanup_pkg, "fetch_ranges", lambda *a, **k: None)
        cleaned, _desc, report = clean(_ad_document(), video_id="abc")
        assert report.sponsorblock == "unavailable"
        # The heuristic still ran.
        assert report.segments_removed == 4
        assert "SponsorBlock unreachable" in " ".join(report.notes)
        assert len(cleaned.segments) == 4

    def test_sponsorblock_miss_is_recorded(self) -> None:
        _cleaned, _desc, report = clean(_ad_document(), video_id="abc")
        assert report.sponsorblock == "miss"
