"""Tests for WQSF quality masking.

These tests pin the three inland-water traps documented in `habforecast.data.qc`. Each
one was a real bug that silently produced an empty dataset or a bloom-biased one, so
each gets a regression test with the flag semantics spelled out.
"""

from __future__ import annotations

import numpy as np
import pytest

from habforecast.data.qc import (
    DEFAULT_REJECT_FLAGS,
    DEFAULT_REQUIRE_ANY,
    FlagTable,
    flag_frequencies,
    valid_water_mask,
)

# Mirrors the real OLCI WQSF table (name order paired with mask values), trimmed to the
# flags these tests exercise. Values are the actual bit masks from the product.
FLAG_MASKS = [1, 2, 4, 8, 4194304, 8388608, 16777216, 16, 32, 64, 8192, 2097152, 33554432]
FLAG_MEANINGS = (
    "INVALID WATER LAND CLOUD TURBID_ATM CLOUD_AMBIGUOUS CLOUD_MARGIN "
    "SNOW_ICE INLAND_WATER COASTLINE SATURATED OCNN_FAIL OC4ME_FAIL"
)


@pytest.fixture
def table() -> FlagTable:
    return FlagTable.from_attrs(np.array(FLAG_MASKS, dtype=np.uint64), FLAG_MEANINGS)


def bits(table: FlagTable, *names: str) -> np.uint64:
    out = np.uint64(0)
    for n in names:
        out |= np.uint64(table.masks[n])
    return out


class TestFlagTable:
    def test_parses_names_and_masks(self, table):
        assert table.masks["INVALID"] == 1
        assert table.masks["INLAND_WATER"] == 32
        assert table.masks["CLOUD_MARGIN"] == 16777216

    def test_rejects_malformed_table(self):
        with pytest.raises(ValueError, match="malformed"):
            FlagTable.from_attrs(np.array([1, 2], dtype=np.uint64), "ONLY_ONE_NAME")

    def test_decodes_bytes_meanings(self):
        t = FlagTable.from_attrs(np.array([1, 2], dtype=np.uint64), b"A B")
        assert t.masks == {"A": 1, "B": 2}

    def test_unknown_flags_are_ignored_not_fatal(self, table):
        # A future processing baseline may drop a flag; masking must not crash.
        assert table.bitmask_for(["INVALID", "NOT_A_REAL_FLAG"]) == np.uint64(1)
        assert table.missing(["INVALID", "NOT_A_REAL_FLAG"]) == ["NOT_A_REAL_FLAG"]


class TestInlandWaterTraps:
    """The three failure modes that made the first ingest return zero valid pixels."""

    def test_land_flag_does_not_disqualify_inland_water(self, table):
        """OLCI sets LAND *and* INLAND_WATER over lakes.

        Rejecting LAND therefore discards every usable lake pixel. Verified on a real
        Lake Erie granule where 75.9% of window pixels carried LAND, 54.9% carried
        INLAND_WATER, and the 40.7% intersection was exactly the set with a finite
        CHL_NN retrieval.
        """
        wqsf = np.array([[bits(table, "LAND", "INLAND_WATER")]], dtype=np.uint64)
        assert valid_water_mask(wqsf, table).all()
        assert "LAND" not in DEFAULT_REJECT_FLAGS

    def test_open_ocean_water_flag_is_not_required(self, table):
        """WATER is never set over lakes; requiring it alone would mask everything."""
        inland_only = np.array([[bits(table, "INLAND_WATER")]], dtype=np.uint64)
        assert valid_water_mask(inland_only, table).all()
        assert "INLAND_WATER" in DEFAULT_REQUIRE_ANY

    def test_bloom_correlated_flags_are_retained(self, table):
        """TURBID_ATM and OC4ME_FAIL correlate with dense blooms.

        Rejecting them would preferentially delete the strongest blooms and bias the
        training set toward clear water -- the opposite of what the model needs.
        """
        blooming = np.array(
            [[bits(table, "INLAND_WATER", "LAND", "TURBID_ATM", "OC4ME_FAIL")]], dtype=np.uint64
        )
        assert valid_water_mask(blooming, table).all()
        assert "TURBID_ATM" not in DEFAULT_REJECT_FLAGS
        assert "OC4ME_FAIL" not in DEFAULT_REJECT_FLAGS


class TestRejection:
    @pytest.mark.parametrize(
        "flag", ["CLOUD", "CLOUD_AMBIGUOUS", "CLOUD_MARGIN", "INVALID", "SNOW_ICE", "OCNN_FAIL"]
    )
    def test_quality_flags_are_rejected(self, table, flag):
        wqsf = np.array([[bits(table, "INLAND_WATER", flag)]], dtype=np.uint64)
        assert not valid_water_mask(wqsf, table).any(), f"{flag} should invalidate the pixel"

    def test_non_water_is_rejected(self, table):
        """Pure land -- no water flag at all -- must never pass."""
        wqsf = np.array([[bits(table, "LAND")]], dtype=np.uint64)
        assert not valid_water_mask(wqsf, table).any()

    def test_mixed_scene_selects_only_good_pixels(self, table):
        wqsf = np.array(
            [
                [bits(table, "INLAND_WATER", "LAND"), bits(table, "INLAND_WATER", "CLOUD")],
                [bits(table, "LAND"), bits(table, "INLAND_WATER", "TURBID_ATM")],
            ],
            dtype=np.uint64,
        )
        expected = np.array([[True, False], [False, True]])
        np.testing.assert_array_equal(valid_water_mask(wqsf, table), expected)

    def test_shape_is_preserved(self, table):
        wqsf = np.full((7, 11), bits(table, "INLAND_WATER"), dtype=np.uint64)
        assert valid_water_mask(wqsf, table).shape == (7, 11)


class TestFlagFrequencies:
    def test_counts_are_fractions(self, table):
        wqsf = np.array(
            [bits(table, "INLAND_WATER"), bits(table, "INLAND_WATER", "CLOUD")], dtype=np.uint64
        )
        freq = flag_frequencies(wqsf, table)
        assert freq["INLAND_WATER"] == pytest.approx(1.0)
        assert freq["CLOUD"] == pytest.approx(0.5)
        assert freq["SNOW_ICE"] == pytest.approx(0.0)

    def test_empty_input_returns_empty(self, table):
        assert flag_frequencies(np.array([], dtype=np.uint64), table) == {}
