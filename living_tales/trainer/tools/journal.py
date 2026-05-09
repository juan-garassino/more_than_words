"""
Living Tales — Auto-filling Journal (J key)
===========================================
Runtime data structure built from emitted scene tuples. The pygame UI toggles
it with J. Sections, per SCHEMA.md:

    1. Locations visited   — unique LOCATION tokens, first-visit turn, count,
                             backdrop thumbnail.
    2. People met          — unique PRESENCE tokens (suspects + witnesses),
                             intro from briefing (suspects) or generic
                             ("a witness"), first-meeting turn, appearances,
                             most recent stance.
    3. Evidence            — OBJECT_FOCUS emitted in a scene where REVELATION
                             also fired. Pairs object → revelation it triggered.
    4. Timeline            — chronological list of REVELATION tokens with the
                             composed scene paragraph (or a fallback summary).

This file ships the headless data model AND a pygame overlay (JournalScreen)
that mirrors pygame_play.py's palette and font set so the J view feels native.

Headless self-test:

    python3 tools/journal.py amber_cipher
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_HERE = Path(__file__).resolve().parent


# ─── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class LocationEntry:
    token_id: str
    display_name: str
    first_visit_turn: int
    visit_count: int = 1
    backdrop_path: Optional[str] = None


@dataclass
class PersonEntry:
    token_id: str           # presence:with_<id>
    name: str
    intro: str
    first_meeting_turn: int
    appearances: int = 1
    most_recent_stance: str = ""


@dataclass
class EvidenceEntry:
    token_id: str
    display_name: str
    discovered_turn: int
    triggered_revelation: str   # the REVELATION token id that fired in the same scene


@dataclass
class TimelineEntry:
    turn: int
    revelation_token: str
    composed_prose: str
    location: str


# ─── Helpers ───────────────────────────────────────────────────────────────────
NONE_TOKENS = {
    "object_focus:none", "object:none",
    "presence:none", "presence:alone",
    "stance:none",
    "revelation:none",
    "location:none",
}


def _phrase(phrases: dict, dim: str, token_id: str, lang: str = "en") -> str:
    block = phrases.get(dim, {})
    entry = block.get(token_id, {})
    if isinstance(entry, dict):
        return entry.get(lang) or entry.get("en") or ""
    return str(entry) if entry else ""


def _humanize(token_id: str) -> str:
    """Fallback display name from a token id like 'presence:with_renard_voss'."""
    if not token_id:
        return ""
    tail = token_id.split(":", 1)[-1]
    for prefix in ("with_", "to_"):
        if tail.startswith(prefix):
            tail = tail[len(prefix):]
    return tail.replace("_", " ").title()


def _suspect_id_from_presence(presence_tok: str) -> Optional[str]:
    """presence:with_renard_voss → suspect:renard_voss (best-effort match)."""
    if not presence_tok or ":" not in presence_tok:
        return None
    tail = presence_tok.split(":", 1)[-1]
    if tail.startswith("with_"):
        return "suspect:" + tail[len("with_"):]
    return None  # noqa: E501


# ─── Composer (lightweight, mirrors SCHEMA.md slot-fill grammar) ──────────────
def compose_scene_prose(scene: dict, phrases: dict, lang: str = "en") -> str:
    """Render a scene-tuple to a paragraph using authored phrases.

    Lightweight version of the composer described in SCHEMA.md. Used as the
    default text for TimelineEntry.composed_prose if the caller doesn't pass
    one through. Skips empty fragments gracefully.
    """
    sentences: List[str] = []

    transition = _phrase(phrases, "TRANSITION", scene.get("TRANSITION", ""), lang)
    location = _phrase(phrases, "LOCATION", scene.get("LOCATION", ""), lang)
    if transition or location:
        s = f"{transition} {location}".strip()
        if s:
            sentences.append(s.rstrip(".") + ".")

    presence_tok = scene.get("PRESENCE", "")
    presence = _phrase(phrases, "PRESENCE", presence_tok, lang)
    stance = _phrase(phrases, "STANCE", scene.get("STANCE", ""), lang)
    if presence and presence_tok not in ("presence:alone", "presence:none"):
        s = presence + (f", {stance}" if stance else "") + "."
        sentences.append(s)
    elif presence_tok == "presence:alone":
        sentences.append(presence.capitalize() + "." if presence else "")

    action = _phrase(phrases, "ACTION", scene.get("ACTION", ""), lang)
    obj = _phrase(phrases, "OBJECT_FOCUS", scene.get("OBJECT_FOCUS", ""), lang)
    if action and obj:
        sentences.append(f"The detective {action} {obj}.")
    elif action:
        sentences.append(f"The detective {action}.")

    tell = _phrase(phrases, "TELL", scene.get("TELL", ""), lang)
    if tell and presence_tok not in ("presence:alone", "presence:none"):
        sentences.append(tell.capitalize().rstrip(".") + ".")

    atmos = _phrase(phrases, "ATMOSPHERE", scene.get("ATMOSPHERE", ""), lang)
    if atmos:
        sentences.append(atmos if atmos.endswith(".") else atmos + ".")

    rev = _phrase(phrases, "REVELATION", scene.get("REVELATION", ""), lang)
    if rev:
        sentences.append(rev if rev.endswith(".") else rev + ".")

    return " ".join(s for s in sentences if s).strip()


# ─── Journal ───────────────────────────────────────────────────────────────────
class Journal:
    """Auto-fills from emitted scenes. View on dialogue history; no extra state."""

    def __init__(self, case_id: str, project_root: Path, lang: str = "en"):
        self.case_id = case_id
        self.project_root = Path(project_root)
        self.lang = lang

        cases_dir = self.project_root / "living_tales" / "trainer" / "cases" / case_id
        # phrases.json
        phrases_p = cases_dir / "phrases.json"
        self.phrases: dict = {}
        if phrases_p.exists():
            with open(phrases_p) as f:
                self.phrases = json.load(f)

        # dimensions.json (declarative dim listing) — kept for completeness
        dims_p = cases_dir / "dimensions.json"
        self.dimensions: dict = {}
        if dims_p.exists():
            try:
                with open(dims_p) as f:
                    self.dimensions = json.load(f)
            except Exception:
                pass

        # scene_map for backdrop thumbnails
        self.scene_map: dict = {}
        sm_p = cases_dir / "scene_map.json"
        if sm_p.exists():
            with open(sm_p) as f:
                self.scene_map = json.load(f)
        art_dir_rel = self.scene_map.get(
            "_art_dir", f"art/{case_id}/direct_pixel_art_v1/pixel_320x320")
        self.art_dir = self.project_root / art_dir_rel

        # case briefing — for People intros
        self.briefing: dict = {}
        for p in (self.project_root / "cases" / f"{case_id}.json",
                  cases_dir.parent / f"{case_id}.json"):
            if p.exists():
                try:
                    with open(p) as f:
                        case_data = json.load(f)
                    self.briefing = case_data.get("briefing", {})
                except Exception:
                    pass
                break

        # Build suspect_id → intro lookup from the briefing (lang fallback en).
        self._intro_by_suspect_id: Dict[str, Dict[str, str]] = {}
        for L in ("en", "es", "fr"):
            block = self.briefing.get(L, {})
            for s in block.get("suspects", []) or []:
                sid = s.get("id")
                if sid:
                    self._intro_by_suspect_id.setdefault(sid, {})[L] = s.get("intro", "")

        # State
        self.locations: Dict[str, LocationEntry] = {}
        self.people: Dict[str, PersonEntry] = {}
        self.evidence: List[EvidenceEntry] = []
        self.timeline: List[TimelineEntry] = []

    # ── Update API ────────────────────────────────────────────────────────────
    def update_from_scene(self, player_card: str, scene: dict, turn: int,
                          composed_prose: Optional[str] = None) -> None:
        # Locations ──────────────────────────────────────────────────────────
        loc_tok = scene.get("LOCATION", "")
        if loc_tok and loc_tok not in NONE_TOKENS:
            entry = self.locations.get(loc_tok)
            if entry is None:
                fname = self.scene_map.get(loc_tok)
                bp: Optional[str] = None
                if fname:
                    candidate = self.art_dir / fname
                    if candidate.exists():
                        bp = str(candidate)
                self.locations[loc_tok] = LocationEntry(
                    token_id=loc_tok,
                    display_name=_phrase(self.phrases, "LOCATION", loc_tok, self.lang)
                                 or _humanize(loc_tok),
                    first_visit_turn=turn,
                    visit_count=1,
                    backdrop_path=bp,
                )
            else:
                entry.visit_count += 1

        # People ─────────────────────────────────────────────────────────────
        pres_tok = scene.get("PRESENCE", "")
        if pres_tok and pres_tok not in ("presence:alone", "presence:none", ""):
            stance_tok = scene.get("STANCE", "") or ""
            stance_phrase = _phrase(self.phrases, "STANCE", stance_tok, self.lang)
            stance_label = stance_phrase or stance_tok.split(":", 1)[-1].replace("_", " ")
            entry = self.people.get(pres_tok)
            if entry is None:
                # Look up suspect briefing intro if this PRESENCE refers to a suspect.
                sid = _suspect_id_from_presence(pres_tok)
                intro = ""
                if sid and sid in self._intro_by_suspect_id:
                    intros = self._intro_by_suspect_id[sid]
                    intro = intros.get(self.lang) or intros.get("en") or ""
                # Fallback intro for witnesses.
                name = _humanize(pres_tok)
                if not intro:
                    # Use the presence phrase as a placeholder intro (one short line).
                    intro = _phrase(self.phrases, "PRESENCE", pres_tok, self.lang) \
                            or f"A witness encountered during the case."
                self.people[pres_tok] = PersonEntry(
                    token_id=pres_tok,
                    name=name,
                    intro=intro,
                    first_meeting_turn=turn,
                    appearances=1,
                    most_recent_stance=stance_label.strip() if stance_tok not in NONE_TOKENS else "",
                )
            else:
                entry.appearances += 1
                if stance_tok and stance_tok not in NONE_TOKENS:
                    entry.most_recent_stance = stance_label.strip()

        # Evidence ───────────────────────────────────────────────────────────
        rev_tok = scene.get("REVELATION", "")
        obj_tok = scene.get("OBJECT_FOCUS", "")
        rev_meaningful = rev_tok and rev_tok not in NONE_TOKENS
        obj_meaningful = obj_tok and obj_tok not in NONE_TOKENS
        if rev_meaningful and obj_meaningful:
            self.evidence.append(EvidenceEntry(
                token_id=obj_tok,
                display_name=_phrase(self.phrases, "OBJECT_FOCUS", obj_tok, self.lang)
                             or _humanize(obj_tok),
                discovered_turn=turn,
                triggered_revelation=rev_tok,
            ))

        # Timeline ───────────────────────────────────────────────────────────
        if rev_meaningful:
            prose = composed_prose
            if not prose:
                prose = compose_scene_prose(scene, self.phrases, self.lang)
            loc_label = _phrase(self.phrases, "LOCATION", loc_tok, self.lang) \
                        or _humanize(loc_tok)
            self.timeline.append(TimelineEntry(
                turn=turn,
                revelation_token=rev_tok,
                composed_prose=prose,
                location=loc_label,
            ))

    # ── Render data ───────────────────────────────────────────────────────────
    def to_render_data(self) -> dict:
        return {
            "locations": sorted(self.locations.values(),
                                key=lambda e: e.first_visit_turn),
            "people": sorted(self.people.values(),
                             key=lambda e: e.first_meeting_turn),
            "evidence": list(self.evidence),  # already in turn order
            "timeline": list(self.timeline),  # already in turn order
        }

    # ── Text dump (for headless preview) ──────────────────────────────────────
    def to_text(self, max_timeline: int = 12) -> str:
        d = self.to_render_data()
        out: List[str] = []
        out.append("═══ JOURNAL ═══")
        out.append("")
        out.append(f"LOCATIONS ({len(d['locations'])})")
        for e in d["locations"]:
            thumb = " [art]" if e.backdrop_path else ""
            out.append(f"  • {e.display_name}  (first turn {e.first_visit_turn}, "
                       f"{e.visit_count}× visited){thumb}")
        out.append("")
        out.append(f"PEOPLE ({len(d['people'])})")
        for e in d["people"]:
            stance = f" — {e.most_recent_stance}" if e.most_recent_stance else ""
            intro = e.intro
            if len(intro) > 80:
                intro = intro[:77] + "…"
            out.append(f"  • {e.name}{stance}")
            out.append(f"      met turn {e.first_meeting_turn} · "
                       f"{e.appearances}× present")
            out.append(f"      {intro}")
        out.append("")
        out.append(f"EVIDENCE ({len(d['evidence'])})")
        for e in d["evidence"]:
            rev = e.triggered_revelation.split(":", 1)[-1].replace("_", " ")
            out.append(f"  • turn {e.discovered_turn:>2}: "
                       f"{e.display_name} → {rev}")
        out.append("")
        out.append(f"TIMELINE ({len(d['timeline'])})")
        for e in d["timeline"][:max_timeline]:
            rev = e.revelation_token.split(":", 1)[-1].replace("_", " ")
            prose = e.composed_prose
            if len(prose) > 120:
                prose = prose[:117] + "…"
            out.append(f"  [{e.turn:>2}] {rev:24s}  @ {e.location}")
            out.append(f"        {prose}")
        if len(d["timeline"]) > max_timeline:
            out.append(f"  …and {len(d['timeline']) - max_timeline} more entries")
        return "\n".join(out)


# ─── Pygame overlay ────────────────────────────────────────────────────────────
class JournalScreen:
    """Pygame overlay for the J-key journal view.

    Mirrors pygame_play.py's palette/fonts so the overlay reads as native
    parchment-on-ink. Four sections in a vertical layout:
        Locations · People · Evidence · Timeline (scrollable)

    Press J or Esc to close. Up/Down arrows scroll the timeline section.
    """

    def __init__(self, screen, fonts: dict, palette: dict):
        # Lazy import — pygame is only needed when rendering, not for the
        # headless data model used in tests / cartridge tooling.
        import pygame  # noqa: F401
        self.screen = screen
        self.fonts = fonts
        self.palette = palette
        self.scroll = 0          # timeline scroll offset (in entries)
        self.timeline_visible = 6
        self._thumb_cache: Dict[str, "pygame.Surface"] = {}

    # ── Event handling ────────────────────────────────────────────────────────
    def handle_event(self, event) -> Optional[str]:
        import pygame
        if event.type != pygame.KEYDOWN:
            return None
        if event.key in (pygame.K_j, pygame.K_ESCAPE):
            return "close"
        if event.key == pygame.K_UP:
            self.scroll = max(0, self.scroll - 1)
            return "scroll_up"
        if event.key == pygame.K_DOWN:
            self.scroll += 1
            return "scroll_down"
        if event.key == pygame.K_PAGEUP:
            self.scroll = max(0, self.scroll - self.timeline_visible)
            return "scroll_up"
        if event.key == pygame.K_PAGEDOWN:
            self.scroll += self.timeline_visible
            return "scroll_down"
        return None

    # ── Rendering ─────────────────────────────────────────────────────────────
    def _wrap(self, font, text: str, max_w: int) -> List[str]:
        words = text.split()
        lines, cur = [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            if font.size(trial)[0] <= max_w:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    def _load_thumb(self, path: Optional[str], size: int = 64):
        import pygame
        if not path:
            return None
        if path in self._thumb_cache:
            return self._thumb_cache[path]
        try:
            img = pygame.image.load(path).convert()
            surf = pygame.transform.scale(img, (size, size))
            self._thumb_cache[path] = surf
            return surf
        except Exception:
            return None

    def render(self, journal: Journal) -> None:
        import pygame
        P = self.palette
        F = self.fonts
        W, H = self.screen.get_size()

        # Parchment veil over backdrop.
        veil = pygame.Surface((W, H), pygame.SRCALPHA)
        veil.fill((*P.get("parchment", (28, 22, 16))[:3], 235))
        self.screen.blit(veil, (0, 0))

        margin = 18
        # Column split for compact layouts: left ~ locations/people, right
        # evidence/timeline. For very narrow screens (<420 px) stack vertically.
        narrow = W < 460
        col_w = (W - margin * 3) // 2 if not narrow else (W - margin * 2)
        col_x_l = margin
        col_x_r = margin if narrow else margin * 2 + col_w

        # ── Header ────────────────────────────────────────────────────────────
        title_s = F["heading"].render("JOURNAL", True, P.get("accent", (196, 150, 64)))
        self.screen.blit(title_s, (margin, 12))
        sub_s = F["small"].render(
            "J / Esc to close   ·   ↑/↓ scroll timeline",
            True, P.get("dim", (150, 140, 116)))
        self.screen.blit(sub_s, (margin, 12 + title_s.get_height() + 2))
        # Gold underline
        pygame.draw.line(
            self.screen, P.get("gold_dim", (124, 96, 36)),
            (margin, 50), (W - margin, 50), 1)

        y_l = 60
        y_r = 60 if not narrow else None

        # ── Locations ─────────────────────────────────────────────────────────
        y_l = self._render_locations(journal, col_x_l, y_l, col_w)
        y_l += 8

        # ── People ────────────────────────────────────────────────────────────
        y_l = self._render_people(journal, col_x_l, y_l, col_w)

        # ── Evidence ──────────────────────────────────────────────────────────
        if narrow:
            y = y_l + 12
        else:
            y = 60
        y = self._render_evidence(journal, col_x_r, y, col_w)
        y += 8

        # ── Timeline (scrollable) ─────────────────────────────────────────────
        self._render_timeline(journal, col_x_r, y, col_w, H - y - margin)

        pygame.display.flip()

    def _section_header(self, text: str, x: int, y: int, w: int) -> int:
        import pygame
        P = self.palette
        F = self.fonts
        s = F["heading"].render(text, True, P.get("accent", (196, 150, 64)))
        self.screen.blit(s, (x, y))
        y += s.get_height() + 2
        pygame.draw.line(self.screen, P.get("gold_dim", (124, 96, 36)),
                         (x, y), (x + w, y), 1)
        y += 6
        return y

    def _render_locations(self, journal, x, y, w) -> int:
        F = self.fonts
        P = self.palette
        d = journal.to_render_data()
        y = self._section_header(f"LOCATIONS ({len(d['locations'])})", x, y, w)
        for entry in d["locations"][:6]:
            thumb = self._load_thumb(entry.backdrop_path, size=48)
            tx = x
            if thumb:
                self.screen.blit(thumb, (tx, y))
                text_x = tx + 56
            else:
                text_x = tx
            name_s = F["body"].render(entry.display_name, True,
                                       P.get("fg", (232, 226, 208)))
            self.screen.blit(name_s, (text_x, y))
            meta = f"first turn {entry.first_visit_turn}  ·  "\
                   f"{entry.visit_count}× visited"
            meta_s = F["small"].render(meta, True, P.get("dim", (150, 140, 116)))
            self.screen.blit(meta_s, (text_x, y + name_s.get_height() + 2))
            y += max(54, name_s.get_height() + meta_s.get_height() + 8)
        return y

    def _render_people(self, journal, x, y, w) -> int:
        F = self.fonts
        P = self.palette
        d = journal.to_render_data()
        y = self._section_header(f"PEOPLE ({len(d['people'])})", x, y, w)
        for entry in d["people"][:5]:
            line1 = entry.name
            if entry.most_recent_stance:
                line1 += f" — {entry.most_recent_stance}"
            n_s = F["body"].render(line1, True, P.get("fg", (232, 226, 208)))
            self.screen.blit(n_s, (x, y))
            y += n_s.get_height() + 1
            meta = f"met turn {entry.first_meeting_turn}  ·  "\
                   f"{entry.appearances}× present"
            m_s = F["small"].render(meta, True, P.get("dim", (150, 140, 116)))
            self.screen.blit(m_s, (x, y))
            y += m_s.get_height() + 1
            # Truncate intro to two lines.
            for line in self._wrap(F["small"], entry.intro, w)[:2]:
                ln_s = F["small"].render(line, True, P.get("sepia", (188, 152, 108)))
                self.screen.blit(ln_s, (x, y))
                y += ln_s.get_height()
            y += 6
        return y

    def _render_evidence(self, journal, x, y, w) -> int:
        F = self.fonts
        P = self.palette
        d = journal.to_render_data()
        y = self._section_header(f"EVIDENCE ({len(d['evidence'])})", x, y, w)
        for entry in d["evidence"][:8]:
            rev = entry.triggered_revelation.split(":", 1)[-1].replace("_", " ")
            line = f"[{entry.discovered_turn:>2}] {entry.display_name} → {rev}"
            for ln in self._wrap(F["small"], line, w)[:2]:
                s = F["small"].render(ln, True, P.get("fg", (232, 226, 208)))
                self.screen.blit(s, (x, y))
                y += s.get_height()
            y += 2
        return y

    def _render_timeline(self, journal, x, y, w, h) -> int:
        import pygame
        F = self.fonts
        P = self.palette
        d = journal.to_render_data()
        y = self._section_header(f"TIMELINE ({len(d['timeline'])})", x, y, w)

        line_h = F["small"].get_linesize()
        # Estimate entries that fit (each takes ~3 lines).
        per_entry = line_h * 3 + 4
        max_visible = max(3, h // per_entry)
        self.timeline_visible = max_visible
        total = len(d["timeline"])
        if self.scroll > max(0, total - max_visible):
            self.scroll = max(0, total - max_visible)
        items = d["timeline"][self.scroll : self.scroll + max_visible]

        for entry in items:
            rev = entry.revelation_token.split(":", 1)[-1].replace("_", " ")
            head = f"[{entry.turn:>2}] {rev}  @ {entry.location}"
            h_s = F["small"].render(head, True, P.get("accent", (196, 150, 64)))
            self.screen.blit(h_s, (x, y))
            y += h_s.get_height()
            for ln in self._wrap(F["small"], entry.composed_prose, w)[:2]:
                ln_s = F["small"].render(ln, True, P.get("fg", (232, 226, 208)))
                self.screen.blit(ln_s, (x, y))
                y += ln_s.get_height()
            y += 4
            if y >= self.screen.get_height() - 12:
                break

        # Scroll indicator.
        if total > max_visible:
            tag = f"{self.scroll + 1}-{min(total, self.scroll + max_visible)} / {total}"
            s = F["small"].render(tag, True, P.get("dim", (150, 140, 116)))
            self.screen.blit(s, (x + w - s.get_width(), y))
        return y


# ─── Self-test ────────────────────────────────────────────────────────────────
def _self_test(case_id: str = "amber_cipher",
               trajectory_id: str = "voss_via_cufflink") -> int:
    project_root = _HERE.parent.parent.parent  # 010-more-than-words/
    cases_dir = _HERE.parent / "cases" / case_id
    traj_p = cases_dir / "trajectories" / f"{trajectory_id}.json"
    if not traj_p.exists():
        print(f"[journal] missing trajectory: {traj_p}", file=sys.stderr)
        return 2

    with open(traj_p) as f:
        traj = json.load(f)

    j = Journal(case_id, project_root, lang="en")

    # Replay turns. Trajectory turn numbers are 1-indexed.
    for t in traj.get("turns", []):
        j.update_from_scene(
            player_card=t.get("player_card", ""),
            scene=t.get("scene", {}),
            turn=int(t.get("turn", 0)),
        )

    print(j.to_text(max_timeline=15))

    d = j.to_render_data()
    n_loc = len(d["locations"])
    n_ppl = len(d["people"])
    n_evi = len(d["evidence"])
    n_tl = len(d["timeline"])

    print()
    print(f"[counts] locations={n_loc}  people={n_ppl}  "
          f"evidence={n_evi}  timeline={n_tl}")

    assert n_loc >= 5,  f"expected ≥5 locations, got {n_loc}"
    assert n_ppl >= 6,  f"expected ≥6 people, got {n_ppl}"
    assert n_evi >= 8,  f"expected ≥8 evidence entries, got {n_evi}"
    assert n_tl  >= 10, f"expected ≥10 timeline entries, got {n_tl}"

    print("[journal] self-test OK.")
    return 0


def main() -> int:
    case_id = sys.argv[1] if len(sys.argv) > 1 else "amber_cipher"
    return _self_test(case_id)


if __name__ == "__main__":
    sys.exit(main())
