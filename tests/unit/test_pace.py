"""PACE: the gate and the resolution argument, both of which are easy to get wrong.

PACE is the instrument anyone would reach for when asked to improve an ocean-colour
forecaster, and both reasons it is not the default here are the kind that get lost in a
refactor:

* the download is credential-gated, and the *unauthenticated* failure is a redirect to a
  login page -- which reads as a network fault unless the code says otherwise;
* its mapped products are 4 km at best, and this project has already measured that a 4 km
  rung has no useful skill on lakes.

No network: the catalogue call is replaced.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

import pytest

from habforecast.data import pace


class TestCredentialGate:
    def test_a_download_without_a_token_says_what_to_do(self, monkeypatch):
        """Not a network error, and not a silent empty result.

        Both credential forms are cleared: a developer machine with a real Earthdata
        entry in ~/.netrc would otherwise take the fallback and reach the network, which
        this file promises not to do.
        """
        monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
        monkeypatch.setattr(pace, "netrc_login", lambda: None)
        g = pace.Granule(name="x.nc", day=None, url="https://example.invalid/x.nc")
        with pytest.raises(PermissionError, match=re.escape("urs.earthdata.nasa.gov")):
            pace.fetch(g)

    def test_an_expired_token_is_reported_as_a_token_problem(self, monkeypatch):
        monkeypatch.setenv("EARTHDATA_TOKEN", "stale")

        def refuse(url, timeout=None, token=None):
            raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)

        monkeypatch.setattr(pace, "_get", refuse)
        g = pace.Granule(name="x.nc", day=None, url="https://example.invalid/x.nc")
        with pytest.raises(PermissionError, match="reissue"):
            pace.fetch(g)

    def test_a_catalogue_entry_with_no_data_url_is_refused_before_the_network(self, monkeypatch):
        monkeypatch.setenv("EARTHDATA_TOKEN", "present")
        with pytest.raises(ValueError, match="no data URL"):
            pace.fetch(pace.Granule(name="x.nc", day=None, url=None))

    def test_the_token_is_read_from_the_environment_only(self, monkeypatch):
        monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
        assert pace.token() is None
        monkeypatch.setenv("EARTHDATA_TOKEN", "abc")
        assert pace.token() == "abc"

    def test_a_download_with_neither_credential_names_both_ways_to_fix_it(self, monkeypatch):
        monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
        monkeypatch.setattr(pace, "netrc_login", lambda: None)
        g = pace.Granule(name="x.nc", day=None, url="https://example.invalid/x.nc")
        with pytest.raises(PermissionError, match="netrc"):
            pace.fetch(g)


class TestNetrcCredential:
    """The host this runs on authenticates by ~/.netrc, not by bearer token, so the
    fallback is the path that actually gets used -- and a credential must never be
    logged, nor read from anywhere but the user's own file."""

    def test_a_missing_netrc_is_not_an_error(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        assert pace.netrc_login() is None

    def test_a_malformed_netrc_is_ignored_rather_than_raised(self, monkeypatch, tmp_path):
        (tmp_path / ".netrc").write_text("this is not a netrc file {{{\n", encoding="utf-8")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        assert pace.netrc_login() is None

    def test_an_entry_for_another_host_is_not_used_for_earthdata(self, monkeypatch, tmp_path):
        (tmp_path / ".netrc").write_text(
            "machine example.com login someone password secret\n", encoding="utf-8"
        )
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        assert pace.netrc_login() is None

    def test_an_earthdata_entry_is_found(self, monkeypatch, tmp_path):
        (tmp_path / ".netrc").write_text(
            f"machine {pace.URS_HOST} login someone password secret\n", encoding="utf-8"
        )
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        assert pace.netrc_login() == ("someone", "secret")

    def test_the_token_wins_when_both_are_present(self, monkeypatch):
        """One credential per request, and the explicit one is the one to honour."""
        monkeypatch.setenv("EARTHDATA_TOKEN", "bearer")
        monkeypatch.setattr(pace, "netrc_login", lambda: ("someone", "secret"))
        seen = {}

        def capture(url, timeout=None, token=None):
            seen["token"] = token
            return b"payload"

        monkeypatch.setattr(pace, "_get", capture)
        got = pace.fetch(pace.Granule(name="x", day=None, url="https://x/y.nc"))
        assert got == b"payload" and seen["token"] == "bearer"

    def test_have_credential_accepts_either_form(self, monkeypatch):
        monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
        monkeypatch.setattr(pace, "netrc_login", lambda: None)
        assert pace.have_credential() is False
        monkeypatch.setattr(pace, "netrc_login", lambda: ("a", "b"))
        assert pace.have_credential() is True

    def test_the_opener_scopes_the_password_to_the_login_host(self):
        """A DAAC redirect must not be able to harvest the Earthdata password."""
        opener = pace._netrc_opener("someone", "secret")
        handler = next(
            h for h in opener.handlers if isinstance(h, urllib.request.HTTPBasicAuthHandler)
        )
        assert handler.passwd.find_user_password(None, f"https://{pace.URS_HOST}") == (
            "someone",
            "secret",
        )
        assert handler.passwd.find_user_password(None, "https://evil.invalid") == (None, None)


class TestResolutionArgument:
    def test_a_registry_sized_lake_is_not_resolvable_at_any_mapped_rung(self):
        for res in pace.L3M_RESOLUTIONS_KM.values():
            ok, why = pace.usable_for(12.0, res)
            assert ok is False
            assert "300 m OLCI" in why, "the message must point at what does work"

    def test_a_large_sea_is_resolvable(self):
        ok, _ = pace.usable_for(400.0, pace.L3M_KM)
        assert ok is True

    def test_the_finest_mapped_rung_is_the_one_quoted(self):
        assert min(pace.L3M_RESOLUTIONS_KM.values()) == pace.L3M_KM
        assert pace.L2_KM < pace.L3M_KM, "Level-2 is the fine one; the argument depends on it"

    def test_the_threshold_matches_what_the_network_needs(self):
        """Three downsamples; below about sixteen pixels the coarse head has nothing."""
        assert pace.MIN_USEFUL_PIXELS >= 8


class TestCatalogue:
    def test_search_parses_granules_without_a_credential(self, monkeypatch):
        payload = {
            "items": [
                {
                    "umm": {
                        "GranuleUR": "PACE_OCI.20260828.L3m.DAY.BGC.V3_2.4km.NRT.nc",
                        "TemporalExtent": {
                            "RangeDateTime": {"BeginningDateTime": "2026-08-28T00:00:00Z"}
                        },
                        "RelatedUrls": [
                            {"Type": "VIEW RELATED INFORMATION", "URL": "https://x/doc.html"},
                            {"Type": "GET DATA", "URL": "https://x/PACE.nc"},
                        ],
                    }
                }
            ]
        }
        monkeypatch.setattr(pace, "_get", lambda url, **kw: json.dumps(payload).encode())
        got = pace.search("bgc")
        assert len(got) == 1
        assert got[0].url.endswith(".nc")
        assert got[0].day.isoformat() == "2026-08-28"

    def test_a_granule_with_only_documentation_links_carries_no_url(self, monkeypatch):
        payload = {
            "items": [
                {
                    "umm": {
                        "GranuleUR": "x",
                        "RelatedUrls": [{"Type": "VIEW RELATED INFORMATION", "URL": "https://x"}],
                    }
                }
            ]
        }
        monkeypatch.setattr(pace, "_get", lambda url, **kw: json.dumps(payload).encode())
        assert pace.search("bgc")[0].url is None

    def test_every_collection_is_a_pace_oci_short_name(self):
        assert all(v.startswith("PACE_OCI_") for v in pace.COLLECTIONS.values())
