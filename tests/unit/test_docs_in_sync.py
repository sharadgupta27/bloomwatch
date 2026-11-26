"""The documentation describes the app that actually exists.

Docs drift silently: a route is added and never listed, a setting is read that nobody can
discover, a figure is renamed and the README shows a broken image. Each test here fails
on one such drift and says what to update, so keeping the docs current is part of making
the change rather than a separate chore someone has to remember.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
README = REPO / "README.md"
DOCS = [README, *sorted((REPO / "docs").glob("*.md")), REPO / "models" / "README.md"]
CODE_DIRS = ["src", "serving", "pipelines", "scripts", "monitoring"]

# FastAPI's own pages and the static mount are not part of the app's documented surface.
FRAMEWORK_ROUTES = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc", "/static"}


def _norm(path: str) -> str:
    """`/api/forecast/jobs/{job_id}` and `/api/forecast/jobs/{id}` are the same route."""
    return re.sub(r"\{[^}]*\}", "{}", path)


def _readme_section(title: str) -> str:
    text = README.read_text(encoding="utf-8")
    start = text.index(f"## {title}")
    nxt = text.find("\n## ", start + 4)
    return text[start : nxt if nxt != -1 else None]


def test_every_api_route_is_in_the_readme_api_table():
    import serving.api.main as api

    documented = _norm(_readme_section("What you can do with it"))
    missing = sorted(
        route.path
        for route in api.app.routes
        if getattr(route, "path", None)
        and route.path not in FRAMEWORK_ROUTES
        and f"`{_norm(route.path)}" not in documented.replace("GET ", "`").replace("POST ", "`")
        and _norm(route.path) not in documented
    )
    assert not missing, (
        f"routes missing from the README's 'Use the API' table: {missing}. "
        "Add a row for each (README.md, section 'What you can do with it')."
    )


def test_every_setting_the_code_reads_is_documented():
    used: set[str] = set()
    for d in CODE_DIRS:
        for f in (REPO / d).rglob("*.py"):
            used |= set(re.findall(r"\bHAB_[A-Z_]+\b", f.read_text(encoding="utf-8")))
    documented = set(re.findall(r"`(HAB_[A-Z_]+)`", _readme_section("Configuration")))
    missing = sorted(used - documented)
    assert not missing, (
        f"settings read by the code but not in the README's Configuration tables: {missing}"
    )
    stale = sorted(documented - used)
    assert not stale, f"README documents settings the code no longer reads: {stale}"


def _links(doc: Path) -> list[str]:
    text = doc.read_text(encoding="utf-8")
    # Markdown links and images, and HTML src/href attributes; code spans are not links.
    text = re.sub(r"`[^`\n]*`", "", text)
    found = re.findall(r"\]\(([^)\s]+)\)", text) + re.findall(r'(?:src|href)="([^"]+)"', text)
    return [u for u in found if not re.match(r"^(https?:|mailto:|#|/)", u)]


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(REPO)))
def test_relative_links_and_images_resolve(doc):
    broken = [u for u in _links(doc) if not (doc.parent / u.split("#")[0]).exists()]
    assert not broken, f"{doc.relative_to(REPO)} links to files that do not exist: {broken}"


def _slug(heading: str) -> str:
    """GitHub's heading anchor: lower case, punctuation dropped, spaces to hyphens."""
    h = re.sub(r"[`*_]", "", heading.strip().lower())
    h = re.sub(r"[^\w\- ]", "", h)
    return h.replace(" ", "-")


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(REPO)))
def test_in_page_anchors_match_a_heading(doc):
    text = doc.read_text(encoding="utf-8")
    slugs = {_slug(h) for h in re.findall(r"^#{1,6} (.+)$", text, re.M)}
    anchors = set(re.findall(r"\]\(#([^)]+)\)", text))
    broken = sorted(anchors - slugs)
    assert not broken, f"{doc.relative_to(REPO)} has anchors with no heading: {broken}"


def test_every_keyboard_shortcut_is_listed_in_the_about_panel():
    js = (REPO / "serving" / "web" / "app.js").read_text(encoding="utf-8")
    html = (REPO / "serving" / "web" / "app.html").read_text(encoding="utf-8")
    keys = set(re.findall(r"\b([a-z]): '(?:pan|probe|place|circle|rect|poly|measure)'", js))
    keys |= set(re.findall(r"e\.key === '([a-z])'", js))
    listed = {k.lower() for k in re.findall(r"<kbd>([A-Za-z])</kbd>", html)}
    missing = sorted(keys - listed)
    assert keys, "found no shortcuts in app.js -- has the keydown handler moved?"
    assert not missing, f"shortcuts handled in app.js but not listed in the About panel: {missing}"


def test_model_card_route_serves_the_card():
    import serving.api.main as api
    from fastapi.testclient import TestClient

    r = TestClient(api.app).get("/model-card")
    assert r.status_code == 200
    assert "Model card" in r.text
    assert "Dockerfile" not in r.text  # sanity: it is the card, not some other file
    dockerfile = (REPO / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.count("COPY docs/MODEL_CARD.md") == 2, (
        "both serving images must ship the model card, or /model-card 404s in Docker"
    )
