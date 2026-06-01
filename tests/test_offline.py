"""Offline unit tests for sen2venus.

These tests deliberately avoid any network access (no Zenodo / Hugging Face
downloads), so they are safe and fast to run in CI. They cover the pure logic:
package surface, URL/suffix tables, and the v2 band bookkeeping and constructor
validation (which runs before any tacoreader/network call).

Tests that need the actual datasets live as ``examples/test_*.py`` scripts and
are intentionally not collected here.
"""
import re

import pytest

import sen2venus
from sen2venus import (
    Sen2Venus,
    Sen2VenusSite,
    Sen2VenusURLs,
    Sen2VenusV2,
)
from sen2venus.dataset.sen2venus import Sen2VenusSubsetSuffixes

MD5_RE = re.compile(r"^[0-9a-f]{32}$")


def test_package_exposes_public_api():
    assert sen2venus.Sen2Venus is Sen2Venus
    assert sen2venus.Sen2VenusSite is Sen2VenusSite
    assert sen2venus.Sen2VenusURLs is Sen2VenusURLs
    assert sen2venus.Sen2VenusV2 is Sen2VenusV2


def test_version_is_pep440_ish():
    assert re.match(r"^\d+\.\d+\.\d+", sen2venus.__version__)


class TestSen2VenusURLs:
    def test_sites_list_is_nonempty_and_unique(self):
        sites = Sen2VenusURLs().get_sites_list()
        assert len(sites) == len(Sen2VenusURLs.urls)
        assert len(set(sites)) == len(sites)  # no duplicate sites
        assert "SUDOUE-4" in sites

    def test_get_url_returns_url_and_valid_md5(self):
        url, md5 = Sen2VenusURLs().get_url("SUDOUE-4")
        assert "SUDOUE-4" in url
        assert url.startswith("https://")
        assert MD5_RE.match(md5)

    def test_every_entry_has_a_valid_md5(self):
        for url, md5 in Sen2VenusURLs.urls:
            assert url.startswith("https://zenodo.org/"), url
            assert MD5_RE.match(md5), (url, md5)

    def test_unknown_site_raises(self):
        with pytest.raises(ValueError):
            Sen2VenusURLs().get_url("NOT-A-REAL-SITE")


class TestSubsetSuffixes:
    def test_suffixes_are_pt_files(self):
        for attr in ("sen_rgbnir", "sen_rededge", "venus_rgbnir", "venus_rededge"):
            value = getattr(Sen2VenusSubsetSuffixes, attr)
            assert value.endswith(".pt"), (attr, value)


class TestSen2VenusV2BandBookkeeping:
    def test_subsets_are_subset_of_band_orders(self):
        for name, bands in Sen2VenusV2.SUBSETS.items():
            assert set(bands) <= set(Sen2VenusV2.LR_BANDS), name
            assert set(bands) <= set(Sen2VenusV2.HR_BANDS), name

    def test_swir_only_on_lr_side(self):
        # SWIR bands exist in the Sentinel-2 (LR) order but not in the Venus (HR) order.
        assert set(Sen2VenusV2.SWIR_BANDS) <= set(Sen2VenusV2.LR_BANDS)
        assert not (set(Sen2VenusV2.SWIR_BANDS) & set(Sen2VenusV2.HR_BANDS))

    @pytest.mark.parametrize(
        "sample_id,expected",
        [
            ("SUDOUE-4_225_foo", "SUDOUE-4"),
            ("FGMANAUS_1", "FGMANAUS"),
            ("ES-IC3XG_0_1", "ES-IC3XG"),
        ],
    )
    def test_site_of_takes_prefix(self, sample_id, expected):
        assert Sen2VenusV2._site_of(sample_id) == expected


class TestSen2VenusV2Validation:
    """Constructor validation happens before any tacoreader/network call."""

    def test_bad_subset_raises_without_network(self):
        with pytest.raises(NotImplementedError):
            Sen2VenusV2(subset="bogus")

    def test_bad_return_type_raises_without_network(self):
        with pytest.raises(NotImplementedError):
            Sen2VenusV2(return_type="bad")
