"""wait_for_host: an ingest pauses through a local network outage instead of dying."""

from __future__ import annotations

import socket

from habforecast.data import olci


def _fake_dns(monkeypatch, failures: int) -> list[str]:
    calls: list[str] = []

    def getaddrinfo(host, port):
        calls.append(host)
        if len(calls) <= failures:
            raise socket.gaierror(11001, "getaddrinfo failed")
        return [("ok",)]

    monkeypatch.setattr(olci.socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(olci.time, "sleep", lambda s: None)
    return calls


def test_returns_at_once_when_host_resolves(monkeypatch):
    calls = _fake_dns(monkeypatch, failures=0)
    assert olci.wait_for_host("https://planetarycomputer.microsoft.com/api/stac/v1")
    assert calls == ["planetarycomputer.microsoft.com"]


def test_waits_until_host_resolves_again(monkeypatch):
    calls = _fake_dns(monkeypatch, failures=3)
    assert olci.wait_for_host("api.eumetsat.int", max_wait_s=3600, poll_s=60)
    assert len(calls) == 4


def test_gives_up_after_max_wait(monkeypatch):
    _fake_dns(monkeypatch, failures=10**6)
    clock = iter(range(0, 10**6, 60))
    monkeypatch.setattr(olci.time, "monotonic", lambda: next(clock))
    assert not olci.wait_for_host("api.eumetsat.int", max_wait_s=600, poll_s=60)
