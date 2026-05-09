"""
Living Tales — Pygame Renderer
==============================
Visual frontend that wraps the same transformer/graph game loop as
play_dialogue.py, swapping a pixel-art backdrop when the model emits a
LOCATION head.

Usage
-----
    cd living_tales/trainer
    python3 tools/pygame_play.py amber_cipher --lang es

The transformer, graph, reactions, sampling, and convergence logic are
imported unchanged from play_dialogue.py. This file only owns input,
rendering, and turn pacing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pygame
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from core.cartridge import CartridgeSpec
from core.token import Token, TokenAgency, TokenClass, TokenStream

from tools.play_dialogue import (
    HAND_SIZE, RED_HERRING_TAGS, _CLASS_COLORS, _CLASS_ICONS,
    _encode_token, _fallback_engine_pick, _get_hint, _get_reaction,
    _load_model_and_spec, _token_name, _token_narrative,
)


# ─── Structured engine detection / loader ───────────────────────────────────
def _has_structured_schema(case_dir: Path) -> bool:
    """Return True if the case ships a multidimensional schema."""
    return (case_dir / "dimensions.json").exists()


def _load_structured_engine(case_id: str, project_root: Path,
                            trainer_root: Path, lang: str):
    """Load model + composer + constraint mask for the new dimensional engine.

    Returns dict with keys: model, composer, constraint_mask, dim_vocab,
    full_vocab, full_vocab_to_idx, dim_vocab_to_idx, fallback_trajectory.
    On any failure, the missing pieces are None (caller falls back).
    """
    case_dir = trainer_root / "cases" / case_id
    out: dict = {
        "model": None, "composer": None, "constraint_mask": None,
        "dim_vocab": None, "full_vocab": None,
        "full_vocab_to_idx": None, "dim_vocab_to_idx": None,
        "fallback_trajectory": None, "lang": lang,
    }

    # Composer (phrases)
    try:
        from generator.structured_scene_composer import SceneComposer
        out["composer"] = SceneComposer.load(case_id, lang=lang)
    except Exception as e:
        print(f"[pygame] structured composer load failed: {e}", file=sys.stderr)

    # Dim vocab from dimensions.json
    try:
        with open(case_dir / "dimensions.json") as f:
            dims_json = json.load(f)
        dim_vocab = {d["name"]: list(d["vocab"])
                     for d in dims_json.get("dimensions", [])}
        out["dim_vocab"] = dim_vocab
    except Exception as e:
        print(f"[pygame] dimensions.json read failed: {e}", file=sys.stderr)
        return out

    # Constraint mask
    try:
        from generator.constraints_compiler import ConstraintMask
        cpath = case_dir / "constraints.json"
        constraints_json = {}
        if cpath.exists():
            with open(cpath) as f:
                constraints_json = json.load(f)
        out["constraint_mask"] = ConstraintMask(
            constraints_json, dim_vocab, token_class_map={})
    except Exception as e:
        print(f"[pygame] constraint mask load failed: {e}", file=sys.stderr)

    # Fallback trajectory (always loaded so we can read opening even when model missing)
    try:
        from generator.trajectory_loader import TrajectoryLoader
        tloader = TrajectoryLoader(case_id, project_root)
        all_trajs = tloader.load_all()
        if all_trajs:
            out["fallback_trajectory"] = all_trajs[0]
    except Exception as e:
        print(f"[pygame] trajectory loader failed: {e}", file=sys.stderr)

    # Model checkpoint
    ckpt_path = trainer_root / "outputs" / case_id / "structured_scene_model.pt"
    if ckpt_path.exists():
        try:
            from trainer.structured_scene_model import StructuredSceneTransformer
            ckpt = torch.load(str(ckpt_path), map_location="cpu",
                              weights_only=False)
            cfg = ckpt.get("config", {})
            model = StructuredSceneTransformer(
                dim_vocab=ckpt["dim_vocab"],
                full_vocab=ckpt["full_vocab"],
                hidden_dim=cfg.get("hidden_dim", 128),
                n_layers=cfg.get("n_layers", 2),
                n_heads=cfg.get("n_heads", 4),
                max_history=cfg.get("max_history", 80),
            )
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            out["model"] = model
            out["full_vocab"] = ckpt["full_vocab"]
            out["dim_vocab"] = ckpt["dim_vocab"]
            out["full_vocab_to_idx"] = ckpt.get("full_vocab_to_idx") or {
                t: i for i, t in enumerate(ckpt["full_vocab"])
            }
            out["dim_vocab_to_idx"] = ckpt.get("dim_vocab_to_idx") or {
                d: {t: i for i, t in enumerate(toks)}
                for d, toks in ckpt["dim_vocab"].items()
            }
            print(f"[pygame] loaded StructuredSceneTransformer from {ckpt_path}")
        except Exception as e:
            print(f"[pygame] structured model load failed: {e}",
                  file=sys.stderr)
    else:
        print(f"[pygame] no structured_scene_model.pt at {ckpt_path} — "
              "using fallback trajectory replay.", file=sys.stderr)

    # Discovery beats — convergence-threshold scaffolding for closing arc.
    try:
        from generator.discovery_beats import DiscoveryBeats
        out["discovery_beats"] = DiscoveryBeats.load(case_id, project_root)
        if out["discovery_beats"]:
            print(f"[pygame] loaded {len(out['discovery_beats'].beats)} discovery beats")
    except Exception as e:
        print(f"[pygame] discovery beats load failed: {e}", file=sys.stderr)
        out["discovery_beats"] = None

    return out


def _try_load_journal(case_id: str, project_root: Path, lang: str):
    """Lazy-import journal.py. Returns (Journal, JournalScreen) or (None, None)."""
    try:
        from tools.journal import Journal, JournalScreen  # type: ignore
        return Journal(case_id, project_root, lang), JournalScreen
    except Exception as e:
        print(f"[pygame] journal not available: {e}", file=sys.stderr)
        return None, None

# ─── Window (portrait, adaptive to display size) ─────────────────────────────
# Layout: title strip · backdrop (clean) · location subtitle · narration · hand.
# The backdrop is sacred — nothing is drawn on top of it.
#
# Two presets: "full" for desktops with room (H=800, 480² edge-to-edge backdrop)
# and "compact" for laptops where the dock would crop the window (H=720,
# 320² centered backdrop with side margins). Selected at module load by
# probing the display height.

def _compute_layout(force: Optional[str] = None) -> dict:
    """Pick a layout sized to fit the user's display. Falls back to compact."""
    h_avail = 800
    try:
        pygame.display.init()
        info = pygame.display.Info()
        # Reserve menu bar + dock + window chrome.
        h_avail = info.current_h - 120
    except Exception:
        pass

    if force == "full" or (force != "compact" and h_avail >= 820):
        return dict(W=480, H=800,
                    title=32, bd=480, bd_x=0,
                    loc=28, narr=112, hand=148)
    return dict(W=480, H=720,
                title=28, bd=320, bd_x=80,
                loc=24, narr=96, hand=244)

# Defer layout selection until first use so a CLI override can short-circuit it.
_LAY: Optional[dict] = None

def _layout() -> dict:
    global _LAY
    if _LAY is None:
        _LAY = _compute_layout(force=os.environ.get("LIVING_TALES_LAYOUT"))
    return _LAY

def _rects():
    L = _layout()
    W = L["W"]
    title = pygame.Rect(0, 0, W, L["title"])
    y = L["title"]
    backdrop = pygame.Rect(L["bd_x"], y, L["bd"], L["bd"])
    y += L["bd"]
    location = pygame.Rect(0, y, W, L["loc"])
    y += L["loc"]
    narration = pygame.Rect(0, y, W, L["narr"])
    y += L["narr"]
    hand = pygame.Rect(0, y, W, L["hand"])
    return W, L["H"], title, backdrop, location, narration, hand

W = 480              # logical width is fixed; H is layout-dependent
FPS = 60
# Module-level rectangles are populated lazily — see Renderer.__init__.
TITLE_RECT     = pygame.Rect(0, 0, W, 32)
BACKDROP_RECT  = pygame.Rect(0, 32, W, 480)
LOCATION_RECT  = pygame.Rect(0, 512, W, 28)
NARRATION_RECT = pygame.Rect(0, 540, W, 112)
HAND_RECT      = pygame.Rect(0, 652, W, 148)
H              = 800

# Victorian-detective palette: deep ink-blue · brass · sepia · cream
BG          = (12, 14, 22)
INK         = (8, 10, 16)
NIGHT       = (20, 22, 32)
PARCHMENT   = (28, 22, 16)        # warm dark — for narration panel
PARCHMENT_2 = (38, 30, 22)
CARD_BG     = (24, 20, 14)
CARD_HI     = (40, 32, 22)
CREAM       = (232, 226, 208)
FG          = CREAM
DIM         = (150, 140, 116)
SEPIA       = (188, 152, 108)
GOLD        = (196, 150, 64)
GOLD_DIM    = (124, 96, 36)
BRASS       = (170, 132, 60)
ACCENT      = GOLD
PLAYER      = (150, 196, 224)
CLUE        = (180, 220, 196)
RED         = (220, 110, 90)

# Warmer, less neon class colors
CLASS_RGB = {
    "SUSPECT":    (210, 122, 110),
    "MOTIVE":     (188, 140, 196),
    "EVENT":      (216, 184, 110),
    "LOCATION":   (140, 196, 200),
    "OBJECT":     (148, 174, 210),
    "ACTION":     (152, 198, 140),
    "EMOTION":    (210, 154, 188),
    "MODIFIER":   (170, 162, 140),
    "WITNESS":    (170, 210, 196),
    "TIME":       (216, 196, 130),
    "ACCOMPLICE": (210, 122, 110),
}


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _wrap(font, text: str, max_w: int) -> List[str]:
    """Word-wrap a string to a list of lines fitting max_w pixels."""
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


def _blit_text(surf, font, text, pos, color, max_w=None, line_h=None):
    if max_w is None:
        surf.blit(font.render(text, True, color), pos)
        return pos[1] + font.get_linesize()
    y = pos[1]
    for line in _wrap(font, text, max_w):
        surf.blit(font.render(line, True, color), (pos[0], y))
        y += line_h or font.get_linesize()
    return y


# ─── Renderer ────────────────────────────────────────────────────────────────
class Renderer:
    def __init__(self, case_id: str, scene_map: dict, project_root: Path,
                 res: int = 320):
        pygame.init()
        # Apply the adaptive layout to the module-level rects so all draw
        # methods can keep referencing them as global constants.
        global W, H, TITLE_RECT, BACKDROP_RECT, LOCATION_RECT
        global NARRATION_RECT, HAND_RECT
        W, H, TITLE_RECT, BACKDROP_RECT, LOCATION_RECT, NARRATION_RECT, HAND_RECT = _rects()
        pygame.display.set_caption(f"Living Tales — {case_id}")
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()

        # Mono for labels/cards/hints (period-mechanical feel),
        # serif italic for narration & headers (period-literary feel).
        self.font_big    = pygame.font.SysFont("palatino,georgia,times", 22,
                                               italic=True)
        self.font_title  = pygame.font.SysFont("palatino,georgia,times", 16,
                                               italic=True)
        self.font_serif  = pygame.font.SysFont("palatino,georgia,times", 15)
        self.font_serif_i= pygame.font.SysFont("palatino,georgia,times", 14,
                                               italic=True)
        self.font        = pygame.font.SysFont("menlo,monaco,courier", 14)
        self.font_sm     = pygame.font.SysFont("menlo,monaco,courier", 12)
        self.font_card   = pygame.font.SysFont("menlo,monaco,courier", 13,
                                               bold=True)
        self.case_title: str = case_id.replace("_", " ").title()
        self.location_label: Optional[str] = None
        # Narration entries: list of (kind, text)
        # kind ∈ {"transition", "clue", "you", "extra"}
        self.narration: List[Tuple[str, str]] = []

        self.res = res
        self.scene_map = scene_map
        # Override art_dir + filenames if --res differs from scene_map default.
        default_dir = scene_map.get(
            "_art_dir", f"art/{case_id}/direct_pixel_art_v1/pixel_320x320")
        if res != 320:
            default_dir = default_dir.replace("pixel_320x320", f"pixel_{res}x{res}")
        self.art_dir = project_root / default_dir
        self.current_backdrop = self._load(scene_map.get("_briefing", scene_map.get("_default")))
        self.next_backdrop = None
        self.crossfade_t = 1.0  # 1.0 = fully showing current
        self.signal_line: Optional[str] = None

    def _load(self, filename: Optional[str]) -> Optional[pygame.Surface]:
        if not filename:
            return None
        # Adapt filename if --res differs from the scene_map default (320).
        if self.res != 320 and "320x320" in filename:
            filename = filename.replace("320x320", f"{self.res}x{self.res}")
        p = self.art_dir / filename
        if not p.exists():
            print(f"[pygame] missing backdrop: {p}", file=sys.stderr)
            return None
        img = pygame.image.load(str(p)).convert()
        return pygame.transform.scale(img, (BACKDROP_RECT.width, BACKDROP_RECT.height))

    def set_backdrop(self, location_token_id: Optional[str], crossfade=True):
        filename = self.scene_map.get(location_token_id) if location_token_id else None
        if not filename:
            filename = self.scene_map.get("_default")
        new_surf = self._load(filename)
        if new_surf is None or new_surf is self.current_backdrop:
            return
        if crossfade and self.current_backdrop:
            self.next_backdrop = new_surf
            self.crossfade_t = 0.0
            self._animate_crossfade()
        else:
            self.current_backdrop = new_surf

    def _animate_crossfade(self, frames=24):
        for i in range(frames + 1):
            t = i / frames
            self.crossfade_t = t
            self._draw_frame()
            pygame.display.flip()
            self.clock.tick(FPS)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
        self.current_backdrop = self.next_backdrop
        self.next_backdrop = None
        self.crossfade_t = 1.0

    def add_clue(self, role: str, text: str):
        if not text:
            return
        kind = "clue" if role == "CLUE" else "you" if role == "YOU" else "extra"
        self.narration.append((kind, text))
        # Keep last 3 entries — panel can hold ~3 wrapped lines comfortably.
        self.narration = self.narration[-4:]

    def add_transition(self, text: str):
        if not text:
            return
        self.narration.append(("transition", text))
        self.narration = self.narration[-4:]

    def set_signal(self, text: Optional[str]):
        self.signal_line = text

    def set_location(self, label: Optional[str]):
        self.location_label = label

    def reset_narration(self):
        self.narration = []

    def _draw_title_strip(self):
        pygame.draw.rect(self.screen, INK, TITLE_RECT)
        pygame.draw.line(self.screen, GOLD_DIM,
                         (0, TITLE_RECT.bottom - 1),
                         (W, TITLE_RECT.bottom - 1), 1)
        # Left-aligned case title in italic gold
        t = self.font_title.render(self.case_title, True, GOLD)
        cy = TITLE_RECT.y + (TITLE_RECT.height - t.get_height()) // 2
        self.screen.blit(t, (12, cy))
        # Right-aligned atmospheric signal in italic sepia, gracefully truncated
        if self.signal_line:
            avail_w = W - (12 + t.get_width() + 24)
            txt = self.signal_line
            while txt and self.font_serif_i.size(txt + "…")[0] > avail_w:
                txt = txt[:-1]
            label = (txt + "…") if txt != self.signal_line else self.signal_line
            sig = self.font_serif_i.render(label, True, SEPIA)
            sy = TITLE_RECT.y + (TITLE_RECT.height - sig.get_height()) // 2
            self.screen.blit(sig, (W - sig.get_width() - 12, sy))

    def _draw_backdrop(self):
        # Pure image — never overlaid.
        bg_rect = BACKDROP_RECT
        # Fill the row band so sidebars (compact layout) are deep ink.
        pygame.draw.rect(self.screen, INK,
                         pygame.Rect(0, bg_rect.y, W, bg_rect.height))
        if self.current_backdrop:
            self.screen.blit(self.current_backdrop, bg_rect.topleft)
        if self.next_backdrop and self.crossfade_t < 1.0:
            self.next_backdrop.set_alpha(int(255 * self.crossfade_t))
            self.screen.blit(self.next_backdrop, bg_rect.topleft)
            self.next_backdrop.set_alpha(255)
        # Thin gold frame around the picture when it is inset from the edges
        # (compact layout). Treats the picture like a mounted photograph.
        if bg_rect.x > 0:
            pygame.draw.rect(self.screen, GOLD_DIM, bg_rect, 1)

    def _draw_location_bar(self):
        r = LOCATION_RECT
        pygame.draw.rect(self.screen, NIGHT, r)
        # Decorative side rules
        label = self.location_label or ""
        if label:
            text = self.font_serif_i.render(f"— {label} —", True, BRASS)
            cx = W // 2 - text.get_width() // 2
            cy = r.y + (r.height - text.get_height()) // 2
            # side rules
            pygame.draw.line(self.screen, GOLD_DIM,
                             (16, r.y + r.height // 2),
                             (cx - 12, r.y + r.height // 2), 1)
            pygame.draw.line(self.screen, GOLD_DIM,
                             (cx + text.get_width() + 12, r.y + r.height // 2),
                             (W - 16, r.y + r.height // 2), 1)
            self.screen.blit(text, (cx, cy))
        else:
            pygame.draw.line(self.screen, GOLD_DIM,
                             (16, r.y + r.height // 2),
                             (W - 16, r.y + r.height // 2), 1)

    def _draw_narration(self, signal: Optional[str] = None):
        r = NARRATION_RECT
        # Warm dark "parchment" ground
        pygame.draw.rect(self.screen, PARCHMENT, r)
        # Top + bottom hairline rules
        pygame.draw.line(self.screen, GOLD_DIM, (0, r.y), (W, r.y), 1)
        pygame.draw.line(self.screen, GOLD_DIM,
                         (0, r.bottom - 1), (W, r.bottom - 1), 1)
        # Left margin gold rule
        pygame.draw.line(self.screen, GOLD_DIM,
                         (10, r.y + 6), (10, r.bottom - 6), 1)

        margin_l = 22
        margin_r = 16
        max_w = r.width - margin_l - margin_r
        y = r.y + 8
        line_h = self.font_serif.get_linesize()

        def draw_kind(kind: str, text: str, y: int) -> int:
            if kind == "transition":
                font = self.font_serif_i
                color = SEPIA
                text = f"~ {text}"
            elif kind == "clue":
                # "CLUE" prefix in tag color, rest in cream
                tag = self.font_card.render("CLUE", True, CLUE)
                self.screen.blit(tag, (margin_l, y))
                tag_w = tag.get_width() + 8
                lines = _wrap(self.font_serif, text, max_w - tag_w)
                if lines:
                    self.screen.blit(self.font_serif.render(lines[0], True, CREAM),
                                     (margin_l + tag_w, y))
                    y += line_h
                for line in lines[1:]:
                    self.screen.blit(self.font_serif.render(line, True, CREAM),
                                     (margin_l + tag_w, y))
                    y += line_h
                return y + 2
            elif kind == "you":
                tag = self.font_card.render("YOU ", True, PLAYER)
                self.screen.blit(tag, (margin_l, y))
                tag_w = tag.get_width() + 8
                lines = _wrap(self.font_serif, text, max_w - tag_w)
                if lines:
                    self.screen.blit(self.font_serif.render(lines[0], True, CREAM),
                                     (margin_l + tag_w, y))
                    y += line_h
                for line in lines[1:]:
                    self.screen.blit(self.font_serif.render(line, True, CREAM),
                                     (margin_l + tag_w, y))
                    y += line_h
                return y + 2
            else:  # extra
                font = self.font_serif
                color = DIM

            for line in _wrap(font, text, max_w):
                self.screen.blit(font.render(line, True, color),
                                 (margin_l, y))
                y += line_h
            return y + 2

        for kind, text in self.narration:
            if y >= r.bottom - 12:
                break
            y = draw_kind(kind, text, y)

        # signal is rendered on the title strip, not in the narration panel.

    def _draw_hand(self, hand: List[Token], reactions: dict, lang: str):
        pygame.draw.rect(self.screen, INK, HAND_RECT)
        # Top brass rule
        pygame.draw.line(self.screen, GOLD_DIM,
                         (0, HAND_RECT.y), (W, HAND_RECT.y), 1)

        if not hand:
            self.screen.blit(
                self.font_serif_i.render("No tokens available — the trail has paused.",
                                         True, DIM),
                (16, HAND_RECT.y + 10))
            return

        n = min(len(hand), HAND_SIZE)
        # Header: bold "EVIDENCE" + dim instructions (use only the digits
        # actually visible).
        n_visible = min(n, 2)
        digits = "·".join(str(i + 1) for i in range(n_visible))
        h1 = self.font_card.render("EVIDENCE", True, GOLD)
        self.screen.blit(h1, (16, HAND_RECT.y + 6))
        h2 = self.font_sm.render(
            f"press [{digits}] to play   ·   A accuse   ·   Q quit",
            True, DIM)
        self.screen.blit(h2, (16 + h1.get_width() + 12, HAND_RECT.y + 9))

        list_top = HAND_RECT.y + 28
        list_h = HAND_RECT.height - 36
        # Cap visible at 2: each card gets enough room for name + hint
        # without overflow. Indices [1..2] always; further cards rotate in
        # as the player consumes the hand.
        visible = min(n, 2)
        more = n - visible
        row_h = list_h // visible
        for i, tok in enumerate(hand[:visible]):
            cls = tok.token_class.value
            color = CLASS_RGB.get(cls, FG)
            rect = pygame.Rect(8, list_top + i * row_h, W - 16, row_h - 6)
            self.screen.set_clip(rect)
            # Card body — warm dark, faint highlight at top
            pygame.draw.rect(self.screen, CARD_BG, rect)
            pygame.draw.rect(self.screen, CARD_HI, rect, 1)
            # Wide left class-color band (the "evidence tag")
            pygame.draw.rect(self.screen, color, (rect.x, rect.y, 5, rect.height))
            # Tiny gold corner ticks for an evidence-card feel
            for cx, cy in [(rect.x + 2, rect.y + 2),
                           (rect.right - 8, rect.y + 2),
                           (rect.x + 2, rect.bottom - 4),
                           (rect.right - 8, rect.bottom - 4)]:
                pygame.draw.rect(self.screen, GOLD_DIM, (cx, cy, 2, 2))

            # [N] number badge in gold
            num = self.font_card.render(f"{i+1}", True, GOLD)
            badge_w = num.get_width() + 12
            pygame.draw.rect(self.screen, INK,
                             (rect.x + 12, rect.y + 6, badge_w, 18))
            pygame.draw.rect(self.screen, GOLD_DIM,
                             (rect.x + 12, rect.y + 6, badge_w, 18), 1)
            self.screen.blit(num, (rect.x + 12 + (badge_w - num.get_width()) // 2,
                                   rect.y + 7))
            # Class label
            tag = self.font_sm.render(cls, True, color)
            self.screen.blit(tag, (rect.x + 12 + badge_w + 8, rect.y + 9))
            # Token name (serif, cream)
            name = _structured_render_hand_name(tok) if isinstance(tok, _TravelCard) else _token_name(tok)
            ny = rect.y + 28
            for line in _wrap(self.font_serif, name, rect.width - 24)[:1]:
                self.screen.blit(self.font_serif.render(line, True, CREAM),
                                 (rect.x + 14, ny))
                ny += self.font_serif.get_linesize()
            # Hint (italic, sepia)
            react = reactions.get(tok.id, {})
            hint = react.get("hint", _get_hint(tok, lang))
            if hint:
                for line in _wrap(self.font_serif_i, hint, rect.width - 24)[:1]:
                    self.screen.blit(self.font_serif_i.render(line, True, SEPIA),
                                     (rect.x + 14, ny))
                    ny += self.font_serif_i.get_linesize()
            self.screen.set_clip(None)

        if more > 0:
            badge = self.font_sm.render(
                f"+{more} more in hand · refresh by playing", True, BRASS)
            self.screen.blit(
                badge, (W - badge.get_width() - 12, HAND_RECT.bottom - 16))
        # List style: stacked rows, 1 column.
        list_top = HAND_RECT.y + 28
        list_h = HAND_RECT.height - 36
        row_h = list_h // max(n, 1)
        for i, tok in enumerate(hand[:n]):
            cls = tok.token_class.value
            color = CLASS_RGB.get(cls, FG)
            rect = pygame.Rect(8, list_top + i * row_h, W - 16, row_h - 4)
            pygame.draw.rect(self.screen, (22, 26, 36), rect)
            pygame.draw.rect(self.screen, color, rect, 1)
            # Class color stripe at left
            pygame.draw.rect(self.screen, color, (rect.x, rect.y, 4, rect.height))
            # [N] number + class
            num = self.font_card.render(f"[{i+1}]", True, ACCENT)
            self.screen.blit(num, (rect.x + 14, rect.y + 6))
            tag = self.font_sm.render(cls, True, color)
            self.screen.blit(tag, (rect.x + 50, rect.y + 8))
            # Name
            name = _structured_render_hand_name(tok) if isinstance(tok, _TravelCard) else _token_name(tok)
            ny = rect.y + 24
            for line in _wrap(self.font_sm, name, rect.width - 24)[:1]:
                self.screen.blit(self.font_sm.render(line, True, FG),
                                 (rect.x + 14, ny))
                ny += self.font_sm.get_linesize()
            # Hint
            react = reactions.get(tok.id, {})
            hint = react.get("hint", _get_hint(tok, lang))
            if hint:
                for line in _wrap(self.font_sm, hint, rect.width - 24)[:1]:
                    self.screen.blit(self.font_sm.render(line, True, DIM),
                                     (rect.x + 14, ny))
                    ny += self.font_sm.get_linesize()

    def _draw_frame(self, hand: List[Token] = None, reactions: dict = None,
                    lang: str = "en", state=None):
        self.screen.fill(BG)
        self._draw_title_strip()
        self._draw_backdrop()
        self._draw_location_bar()
        self._draw_narration(self.signal_line)
        if hand is not None:
            self._draw_hand(hand, reactions or {}, lang)

    def draw(self, hand, reactions, lang, state=None):
        self._draw_frame(hand, reactions, lang, state)
        pygame.display.flip()

    # ── modal screens ───────────────────────────────────────────────────────
    def _wait_key(self, keys=(pygame.K_SPACE, pygame.K_RETURN)):
        pygame.event.clear()
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if ev.type == pygame.KEYDOWN:
                    if ev.key in keys:
                        return ev.key
                    # Only Esc exits — Q is reserved for the in-game quit
                    # so it can't accidentally kill the briefing flow.
                    if ev.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit(0)
            self.clock.tick(FPS)

    def show_text_screen(self, title: str, body_lines: List[str], prompt="Press SPACE"):
        # Pre-wrap every body line, then paginate by available height.
        # Use the small font for body so briefings fit on fewer pages.
        body_font = self.font_sm
        margin = 24
        top = 60
        bottom_reserved = 48
        if title:
            top += 50
        line_h = body_font.get_linesize() + 2
        para_gap = 6

        # Flatten into a list of (text, is_blank) wrapped fragments.
        fragments: List[str] = []
        for line in body_lines:
            if not line:
                fragments.append("")
                continue
            for wrapped in _wrap(body_font, line, W - 2 * margin):
                fragments.append(wrapped)
            fragments.append("")  # paragraph gap

        # Build pages: each page fills the available height.
        avail = H - top - bottom_reserved
        pages: List[List[str]] = []
        cur, cur_h = [], 0
        for frag in fragments:
            h = (para_gap if frag == "" else line_h)
            if cur_h + h > avail and cur:
                pages.append(cur)
                cur, cur_h = [], 0
            cur.append(frag)
            cur_h += h
        if cur:
            pages.append(cur)
        if not pages:
            pages = [[]]

        for pi, page in enumerate(pages):
            self.set_backdrop(None, crossfade=False)
            self._draw_backdrop()
            veil = pygame.Surface((W, H), pygame.SRCALPHA)
            veil.fill((0, 0, 0, 170))
            self.screen.blit(veil, (0, 0))

            y = 32
            if title and pi == 0:
                self.screen.blit(self.font_big.render(title, True, ACCENT),
                                 (margin, y))
                y = 60 + 30
            else:
                y = 60

            for frag in page:
                if frag == "":
                    y += para_gap
                else:
                    self.screen.blit(body_font.render(frag, True, FG),
                                     (margin, y))
                    y += line_h

            page_label = (f"{prompt}  ({pi + 1}/{len(pages)})"
                          if len(pages) > 1 else prompt)
            self.screen.blit(self.font_sm.render(page_label, True, DIM),
                             (margin, H - 40))
            pygame.display.flip()
            self._wait_key()

    def wait_for_choice(self, n_cards: int) -> str:
        """Return: digit '1'..'7', 'a' (accuse), 'q' (quit)."""
        # Drain any stale events leaked from prior screens to avoid
        # accidental quits / stray accusations when the user mashed SPACE.
        pygame.event.clear()
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return "q"
                if ev.type == pygame.KEYDOWN:
                    if pygame.K_1 <= ev.key <= pygame.K_9:
                        n = ev.key - pygame.K_0
                        if 1 <= n <= n_cards:
                            return str(n)
                    if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                        return "q"
                    if ev.key == pygame.K_a:
                        return "a"
            self.clock.tick(FPS)


# ─── Game state ──────────────────────────────────────────────────────────────
class GameState:
    """Owns the same state dict that game_loop() in play_dialogue maintains."""

    def __init__(self, spec: CartridgeSpec, model, mappings):
        self.spec = spec
        self.model = model
        self.mappings = mappings

        player_tokens = [
            t for t in spec.tokens
            if t.agency in (TokenAgency.PLAYER, TokenAgency.SHARED)
            and not t.is_invariant and t.stream != TokenStream.OPENING
        ]
        engine_tokens = [
            t for t in spec.tokens
            if t.agency in (TokenAgency.ENGINE, TokenAgency.SHARED)
            and not t.is_invariant and t.stream != TokenStream.OPENING
        ]
        # Phase-aware ordering: shuffle first for variety, then sort so EARLY
        # tokens lead. Without this, a random shuffle can hand the player
        # 7 MID/LATE cards while game_turn=1 only allows EARLY — the hand
        # then reads as empty and the loop breaks on turn 1. See
        # is_available_at_turn() in core/token.py:65.
        np.random.shuffle(player_tokens)
        _phase_rank = {"EARLY": 0, "ANY": 0, "MID": 1, "LATE": 2}
        player_tokens.sort(key=lambda t: _phase_rank.get(t.phase.value, 1))
        self.hand = player_tokens[:HAND_SIZE]
        self.deck = player_tokens[HAND_SIZE:]
        self.engine_pool = engine_tokens

        self.convergence_dims = np.zeros(spec.n_attractor_dims, dtype=np.float32)
        self.placed_ids: set = set()
        self.context_ids: List[str] = []
        self.dialogue_history: List[Tuple[str, Token]] = []
        self.seq_t, self.seq_c, self.seq_p, self.seq_s, self.seq_a = [], [], [], [], []
        self.turn = 0
        self.max_turns = spec.max_turns * 2

    def open(self):
        scene_locs = []
        clues = []
        for tid in self.spec.opening_token_ids:
            tok = self.spec.get_token(tid)
            self.placed_ids.add(tok.id)
            self.context_ids.append(tok.id)
            self.convergence_dims = np.minimum(
                1.0, self.convergence_dims + tok.attractor_weights * self.spec.convergence_rate,
            )
            self.dialogue_history.append(("FIELD", tok))
            if self.mappings:
                enc = _encode_token(tok, self.mappings)
                self.seq_t.append(enc[0]); self.seq_c.append(enc[1])
                self.seq_p.append(enc[2]); self.seq_s.append(enc[3]); self.seq_a.append(enc[4])
            self.turn += 1
            if tok.token_class == TokenClass.LOCATION:
                scene_locs.append(tok)
            else:
                clues.append(tok)
        return scene_locs, clues

    def valid_hand(self):
        gt = self.turn // 2
        return [t for t in self.hand if t.is_available_at_turn(gt)]

    def refill_hand(self):
        """Pull phase-eligible cards from the deck into the hand.

        Naive `deck.pop(0)` refill is wrong: it dumps phase-locked cards into
        the hand where they sit useless, while eligible cards remain unreached
        deeper in the deck. Here we *scan* the deck for cards that pass the
        phase gate at the current `game_turn` and pull those preferentially.
        """
        gt = self.turn // 2
        # First pass — pull eligible cards from anywhere in the deck.
        i = 0
        while i < len(self.deck) and len(self.hand) < HAND_SIZE:
            if self.deck[i].is_available_at_turn(gt):
                self.hand.append(self.deck.pop(i))
            else:
                i += 1
        # Second pass — top up with anything left (so hand stays full
        # for upcoming phase unlocks). Non-eligible cards are filtered
        # by valid_hand() at draw time.
        while len(self.hand) < HAND_SIZE and self.deck:
            self.hand.append(self.deck.pop(0))

    def advance_until_eligible(self) -> bool:
        """If neither hand nor deck has a phase-eligible card, jump turn
        forward until one does. Returns True if we advanced (with cards
        becoming eligible), False if no future turn unlocks anything.
        """
        if self.is_over():
            return False
        for jump in range(2, 30, 2):
            future_turn = self.turn + jump
            if future_turn >= self.max_turns:
                return False
            gt_future = future_turn // 2
            pool = self.hand + self.deck
            if any(t.is_available_at_turn(gt_future) for t in pool):
                self.turn = future_turn
                return True
        return False

    def play_card(self, tok: Token):
        self.hand.remove(tok)
        self.placed_ids.add(tok.id)
        self.context_ids.append(tok.id)
        self.convergence_dims = np.minimum(
            1.0, self.convergence_dims + tok.attractor_weights * self.spec.convergence_rate,
        )
        if set(getattr(tok, 'affinity_tags', [])) & RED_HERRING_TAGS:
            self.convergence_dims = np.maximum(
                0.0,
                self.convergence_dims - abs(np.array(tok.attractor_weights))
                * self.spec.convergence_rate * 0.5,
            )
        self.dialogue_history.append(("YOU", tok))
        if self.mappings:
            enc = _encode_token(tok, self.mappings)
            self.seq_t.append(enc[0]); self.seq_c.append(enc[1])
            self.seq_p.append(enc[2]); self.seq_s.append(enc[3]); self.seq_a.append(enc[4])
        self.turn += 1
        # Phase-aware refill so eligible cards are pulled from anywhere
        # in the deck, not just the front.
        self.refill_hand()

    def engine_response(self) -> Tuple[List[Token], np.ndarray]:
        """Run the model (or fallback) and return (scene_tokens, prev_convergence)."""
        prev = self.convergence_dims.copy()
        scene_tokens: List[Token] = []
        gt = self.turn // 2

        if self.model is not None and self.mappings is not None:
            dev = next(self.model.parameters()).device
            inp = lambda x: torch.tensor([x], dtype=torch.long, device=dev)
            inp_t, inp_c, inp_p, inp_s, inp_a = (
                inp(self.seq_t), inp(self.seq_c), inp(self.seq_p),
                inp(self.seq_s), inp(self.seq_a),
            )
            valid_mask = torch.zeros(self.spec.vocab_size, dtype=torch.bool, device=dev)
            id_to_idx = self.mappings["id_to_idx"]
            for t in self.engine_pool:
                if t.id not in self.placed_ids and t.is_available_at_turn(gt):
                    valid_mask[id_to_idx[t.id]] = True
            idx_to_id = {v: k for k, v in id_to_idx.items()}

            if hasattr(self.model, "predict_scene") and valid_mask.any():
                n_heads = self.model.n_output_heads
                per_head_valid = [valid_mask.clone() for _ in range(n_heads)]

                # Graph-driven logit bias (mirrors play_dialogue.py)
                GRAPH_BOOST, REPULSION_PENALTY = 5.0, 3.0
                recent = [t.id for r, t in self.dialogue_history if r == "YOU"][-5:]
                logit_bias = None
                if recent:
                    active = set()
                    for r, t in self.dialogue_history:
                        if r == "YOU":
                            active.update(getattr(t, 'affinity_tags', []))
                    graph = self.spec.token_graph
                    idx_to_id_map = idx_to_id
                    logit_bias = []
                    for d in range(n_heads):
                        bias = torch.zeros(self.spec.vocab_size)
                        for tok_idx in range(self.spec.vocab_size):
                            tid = idx_to_id_map.get(tok_idx, "")
                            aff = sum(graph.weight(tid, p) for p in recent)
                            bias[tok_idx] += aff * GRAPH_BOOST
                            tok_obj = self.spec.get_token(tid) if tid else None
                            if tok_obj and set(getattr(tok_obj, 'repulsion_tags', [])) & active:
                                bias[tok_idx] -= REPULSION_PENALTY
                        logit_bias.append(bias)

                results = self.model.predict_scene(
                    inp_t, inp_c, inp_p, inp_s, inp_a,
                    per_head_valid=per_head_valid, temperature=0.8,
                    logit_bias=logit_bias,
                )
                for chosen_idx, _ in results:
                    if chosen_idx < 0:
                        continue
                    cid = idx_to_id.get(chosen_idx)
                    if cid is None or cid in self.placed_ids:
                        continue
                    tok = self.spec.get_token(cid)
                    scene_tokens.append(tok)
                    self.placed_ids.add(tok.id)
            elif valid_mask.any():
                chosen_idx, _ = self.model.predict_next(
                    inp_t, inp_c, inp_p, inp_s, inp_a,
                    valid_mask=valid_mask, temperature=0.8,
                )
                tok = self.spec.get_token(idx_to_id[chosen_idx])
                scene_tokens.append(tok)
                self.placed_ids.add(tok.id)
        else:
            available = [t for t in self.engine_pool if t.id not in self.placed_ids]
            tok = _fallback_engine_pick(self.spec, available, self.context_ids, gt)
            if tok:
                scene_tokens.append(tok)
                self.placed_ids.add(tok.id)

        # Place scene tokens in state
        for stok in scene_tokens:
            self.context_ids.append(stok.id)
            self.convergence_dims = np.minimum(
                1.0, self.convergence_dims + stok.attractor_weights * self.spec.convergence_rate,
            )
            self.dialogue_history.append(("FIELD", stok))
            if self.mappings:
                enc = _encode_token(stok, self.mappings)
                self.seq_t.append(enc[0]); self.seq_c.append(enc[1])
                self.seq_p.append(enc[2]); self.seq_s.append(enc[3]); self.seq_a.append(enc[4])
        self.turn += 1
        return scene_tokens, prev

    def passive_decay(self):
        decay = 0.005 + self.convergence_dims * 0.01
        self.convergence_dims = np.maximum(0.0, self.convergence_dims - decay)

    def last_player_token(self) -> Optional[Token]:
        for r, t in reversed(self.dialogue_history):
            if r == "YOU":
                return t
        return None

    def is_over(self):
        return self.turn >= self.max_turns


# ─── Structured engine state ─────────────────────────────────────────────────
class StructuredGameState:
    """State for a case using the multidimensional schema.

    Keeps a flat history of (player_card_id, scene_dict) pairs, the set of
    visited locations, ordered list of previous LOCATION token ids, scene
    index and game turn. Hand consists of a mix of inquiry cards (token
    objects from the spec) + travel cards (synthetic Token-like objects).
    """

    def __init__(self, spec: CartridgeSpec, dim_vocab: Dict[str, List[str]],
                 dimensions_json: dict):
        self.spec = spec
        self.dim_vocab = dim_vocab
        self.dimensions_json = dimensions_json

        # Inquiry pool — player/shared, non-invariant, non-opening tokens
        # whose dim is one of OBJECT_FOCUS / PRESENCE / ACTION / MOTIVE
        # equivalents — practically: the same set legacy used.
        inquiry = [
            t for t in spec.tokens
            if t.agency in (TokenAgency.PLAYER, TokenAgency.SHARED)
            and not t.is_invariant
            and t.stream != TokenStream.OPENING
        ]
        np.random.shuffle(inquiry)
        _phase_rank = {"EARLY": 0, "ANY": 0, "MID": 1, "LATE": 2}
        inquiry.sort(key=lambda t: _phase_rank.get(t.phase.value, 1))
        self.inquiry_pool = inquiry

        # Travel cards from dimensions.json (synthetic tokens)
        self.travel_cards: List[_TravelCard] = []
        travel_ids = (dimensions_json.get("player_cards", {})
                      .get("travel", []) or [])
        for tid in travel_ids:
            stem = tid.split(":", 1)[-1]
            target_loc = stem.replace("to_", "location:", 1) \
                if stem.startswith("to_") else None
            self.travel_cards.append(_TravelCard(
                id=tid, target_location=target_loc))

        # Hand: 5 inquiry + 2 travel (capped at total HAND_SIZE)
        self.hand: List[Any] = []
        self.deck: List[Any] = list(inquiry[5:])
        self.hand.extend(inquiry[:5])
        # Add 2 random travel cards
        if self.travel_cards:
            travel_sample = list(self.travel_cards)
            np.random.shuffle(travel_sample)
            self.hand.extend(travel_sample[:2])

        self.history: List[Tuple[str, Dict[str, str]]] = []  # (player_card_id, scene)
        self.opening_tokens: List[str] = []
        self.previous_locations: List[str] = []
        self.visited_locations: set = set()
        self.scene_index: int = 0
        self.game_turn: int = 0
        # Convergence dims kept for compat — most cases keep 3-dim attractor
        self.convergence_dims = np.zeros(spec.n_attractor_dims, dtype=np.float32)
        self.last_player_card_id: Optional[str] = None
        self.last_scene: Optional[Dict[str, str]] = None

    def open(self, opening_token_ids: List[str]):
        self.opening_tokens = list(opening_token_ids)
        for tid in opening_token_ids:
            if tid.startswith("location:"):
                self.previous_locations.append(tid)
                self.visited_locations.add(tid)

    def play_inquiry(self, tok):
        if tok in self.hand:
            self.hand.remove(tok)
        self.last_player_card_id = tok.id
        # Refill from deck
        while len(self.hand) < HAND_SIZE and self.deck:
            self.hand.append(self.deck.pop(0))

    def play_travel(self, card: "_TravelCard"):
        if card in self.hand:
            self.hand.remove(card)
        self.last_player_card_id = card.id
        # Travel cards aren't refilled from inquiry deck — we re-add a fresh
        # random travel card so the player keeps mobility.
        if self.travel_cards:
            available = [c for c in self.travel_cards if c not in self.hand]
            if available:
                self.hand.append(available[np.random.randint(len(available))])

    def commit_scene(self, scene: Dict[str, str]):
        self.history.append((self.last_player_card_id, scene))
        self.last_scene = scene
        loc = scene.get("LOCATION")
        if loc and loc != "location:none":
            if not self.previous_locations or self.previous_locations[-1] != loc:
                self.previous_locations.append(loc)
            self.visited_locations.add(loc)
        self.scene_index += 1
        self.game_turn += 1

    def build_game_state(self) -> Dict[str, Any]:
        return {
            "previous_locations": list(self.previous_locations),
            "visited_locations": set(self.visited_locations),
            "scene_index": self.scene_index,
            "convergence_dims": list(self.convergence_dims),
            "game_turn": self.game_turn + 1,
            "last_player_card": self.last_player_card_id,
        }


class _TravelCard:
    """Lightweight synthetic card for travel:to_<location> entries.

    Mirrors the minimal Token-shaped surface the renderer reads: id, a
    .token_class.value attribute (used for color), and a name accessor.
    """

    class _Cls:
        def __init__(self, value: str):
            self.value = value

    def __init__(self, id: str, target_location: Optional[str]):
        self.id = id
        self.target_location = target_location
        self.token_class = self._Cls("TRAVEL")
        self.phase = self._Cls("ANY")
        self.surface_expression = ""
        self.attractor_weights = np.zeros(3, dtype=np.float32)
        self.affinity_tags: List[str] = []
        self.repulsion_tags: List[str] = []

    def is_available_at_turn(self, turn: int) -> bool:
        return True


# Distinct color for travel cards
CLASS_RGB["TRAVEL"] = (200, 180, 120)


def _structured_render_hand_name(card) -> str:
    if isinstance(card, _TravelCard):
        stem = card.id.split(":", 1)[-1].replace("to_", "")
        return "Go to " + stem.replace("_", " ").title()
    return _token_name(card)


def _structured_run_engine(engine: dict, state: StructuredGameState,
                           player_card_id: str) -> Dict[str, str]:
    """Run StructuredSceneTransformer.predict_scene, or fallback to trajectory."""
    model = engine.get("model")
    composer = engine.get("composer")
    constraint_mask = engine.get("constraint_mask")
    dim_vocab = engine.get("dim_vocab") or {}

    # If no model: replay the matching turn from the fallback trajectory if
    # possible, else synthesize a deterministic stub scene.
    if model is None:
        traj = engine.get("fallback_trajectory")
        if traj is not None and 0 <= state.scene_index < len(traj.turns):
            return dict(traj.turns[state.scene_index].scene)
        # Last-ditch stub scene: stay where we are.
        last_loc = (state.previous_locations[-1]
                    if state.previous_locations else "location:none")
        return {
            "LOCATION": last_loc,
            "TRANSITION": "transition:stayed",
            "CAUSE": "cause:examining_evidence",
            "PRESENCE": "presence:alone",
            "STANCE": "stance:none",
            "ACTION": "action:examines",
            "OBJECT_FOCUS": "object_focus:none",
            "TELL": "tell:none",
            "ATMOSPHERE": "atmosphere:fog_thickens",
            "REVELATION": "revelation:none",
            "BEAT": "beat:orientation",
        }

    # Build history token + dim tensors from prior turns.
    full_to_idx = engine["full_vocab_to_idx"]
    pad_id = model.pad_id
    dim_to_tag = model.dim_to_tag_id
    tokens: List[int] = []
    dims_seq: List[int] = []

    # opening tokens (treated as pad-dim — informational)
    for tid in state.opening_tokens:
        if tid in full_to_idx:
            tokens.append(full_to_idx[tid])
            dims_seq.append(model.dim_pad_id)

    for pcard, scene in state.history:
        if pcard and pcard in full_to_idx:
            tokens.append(full_to_idx[pcard])
            dims_seq.append(model.dim_pad_id)
        for dim in model.DIM_ORDER:
            tok = scene.get(dim)
            if tok and tok in full_to_idx:
                tokens.append(full_to_idx[tok])
                dims_seq.append(dim_to_tag.get(dim, model.dim_pad_id))

    # Trim to max_history from the right.
    if len(tokens) > model.max_history:
        tokens = tokens[-model.max_history:]
        dims_seq = dims_seq[-model.max_history:]
    if not tokens:
        tokens = [pad_id]
        dims_seq = [model.dim_pad_id]

    pcard_idx = full_to_idx.get(player_card_id, pad_id)
    history_tensors = {
        "tokens": torch.tensor(tokens, dtype=torch.long),
        "dims": torch.tensor(dims_seq, dtype=torch.long),
    }

    try:
        scene = model.predict_scene(
            history=history_tensors,
            player_card=pcard_idx,
            constraint_mask=constraint_mask,
            game_state=state.build_game_state(),
            temperature=0.1,
        )
        return scene
    except Exception as e:
        print(f"[pygame] predict_scene failed: {e}", file=sys.stderr)
        traj = engine.get("fallback_trajectory")
        if traj is not None and 0 <= state.scene_index < len(traj.turns):
            return dict(traj.turns[state.scene_index].scene)
        return {d: dim_vocab.get(d, ["none"])[-1] for d in model.DIM_ORDER}


def _structured_compose(engine: dict, scene: Dict[str, str],
                        last_player_card_id: Optional[str]) -> str:
    """Compose prose; on failure return a brief readable fallback."""
    composer = engine.get("composer")
    if composer is not None:
        try:
            return composer.compose(scene)
        except Exception as e:
            print(f"[pygame] composer.compose failed: {e}", file=sys.stderr)
    # Fallback: glue the dims into a one-liner.
    pieces = []
    for d in ("TRANSITION", "LOCATION", "ACTION", "OBJECT_FOCUS",
              "REVELATION"):
        v = scene.get(d, "")
        if v and not v.endswith(":none"):
            pieces.append(v.split(":", 1)[-1].replace("_", " "))
    return ". ".join(pieces).capitalize() + "."


def _structured_run(case_id: str, lang: str, res: int,
                    project_root: Path, trainer_root: Path,
                    spec: CartridgeSpec, scene_map: dict, case_data: dict,
                    correct_id: Optional[str]):
    """Pygame loop for the multidimensional structured engine."""
    engine = _load_structured_engine(case_id, project_root, trainer_root, lang)

    # Journal — lazy import. None if module missing.
    journal, journal_screen_cls = _try_load_journal(
        case_id, project_root, lang)

    renderer = Renderer(case_id, scene_map, project_root, res=res)
    state = StructuredGameState(spec, engine["dim_vocab"] or {},
                                {"player_cards": engine.get("dim_vocab")
                                                 and {} or {}})
    # Ensure travel cards loaded — pull them from full dimensions.json
    try:
        with open(trainer_root / "cases" / case_id / "dimensions.json") as f:
            dims_full = json.load(f)
        state = StructuredGameState(spec, engine["dim_vocab"] or {}, dims_full)
    except Exception:
        pass

    # Title + briefing
    renderer.show_text_screen(spec.title, [
        "A symbolic mystery." if lang == "en" else "Un misterio simbólico.",
        "",
        "Play tokens — the engine reconstructs the scene." if lang == "en"
        else "Juega fichas — el motor reconstruye la escena.",
    ])
    renderer.show_text_screen("", _briefing_lines(case_data, lang))
    renderer.case_title = spec.title

    # Open scene from spec opening tokens
    state.open(spec.opening_token_ids)
    if state.previous_locations:
        first_loc = state.previous_locations[0]
        renderer.set_backdrop(first_loc, crossfade=True)
        renderer.set_location(first_loc.split(":", 1)[-1]
                               .replace("_", " ").title())

    accused = None
    quit_requested = False

    while True:
        valid = list(state.hand)
        if not valid:
            break

        renderer.draw(valid, {}, lang, state)
        choice = _structured_wait_for_choice(
            renderer, len(valid), engine, state, journal, journal_screen_cls,
            lang)

        if choice == "q":
            quit_requested = True
            break
        if choice == "a":
            picked = _accuse(renderer, spec)
            if picked is not None:
                accused = picked
                break
            continue
        if choice == "j":
            # Journal already handled inside wait_for_choice; just redraw.
            continue

        try:
            idx = int(choice) - 1
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(valid):
            continue
        card = valid[idx]

        renderer.reset_narration()
        renderer.add_clue("YOU", _structured_render_hand_name(card))

        # Apply the card to state
        if isinstance(card, _TravelCard):
            state.play_travel(card)
        else:
            state.play_inquiry(card)
        renderer.draw(state.hand, {}, lang, state)
        pygame.time.wait(200)

        # Run engine
        scene = _structured_run_engine(engine, state,
                                        state.last_player_card_id or "")
        # Player-card → scene binding. The model has not learned this
        # binding strongly enough on its own (subagent judge flagged:
        # "player plays coal_dust, scene narrates the telegram"). Always
        # honor the played card as the scene's focal token.
        played = state.last_player_card_id or ""
        pc_meta = next(
            (t for t in engine["spec"].get("tokens", []) if t.get("id") == played),
            {},
        )
        pc_class = (pc_meta.get("token_class")
                    or pc_meta.get("class") or "").upper()
        dims_by_name = {d["name"]: d["vocab"]
                        for d in engine["dimensions"]["dimensions"]}

        if isinstance(card, _TravelCard) and card.target_location:
            # Travel-card override: force LOCATION + TRANSITION dims
            scene["LOCATION"] = card.target_location
            prev_loc = (state.previous_locations[-1]
                        if state.previous_locations else None)
            if prev_loc == card.target_location:
                scene["TRANSITION"] = "transition:stayed"
            else:
                scene["TRANSITION"] = "transition:crossed_to"
        elif pc_class in ("OBJECT", "MODIFIER"):
            if played in dims_by_name.get("OBJECT_FOCUS", []):
                scene["OBJECT_FOCUS"] = played
                if scene.get("ACTION") in (
                    "action:arrives", "action:leaves",
                    "action:none", None,
                ):
                    scene["ACTION"] = "action:examines"
        elif pc_class in ("SUSPECT", "WITNESS"):
            presence_id = f"presence:with_{played.split(':', 1)[-1]}"
            if presence_id in dims_by_name.get("PRESENCE", []):
                scene["PRESENCE"] = presence_id
                if scene.get("ACTION") in (
                    "action:arrives", "action:leaves",
                    "action:none", None,
                ):
                    scene["ACTION"] = "action:questions"
        elif pc_class in ("MOTIVE", "EVENT", "EMOTION", "ACTION", "TIME"):
            # Player is recalling/connecting an abstract — bias ACTION,
            # clear stale OBJECT_FOCUS so we don't get verb-object
            # leakage like "questioned the coal dust".
            if scene.get("ACTION") in (
                "action:arrives", "action:leaves", "action:none",
                "action:waits", "action:questions",
                "action:examines", None,
            ):
                scene["ACTION"] = "action:recalls"
            scene["OBJECT_FOCUS"] = "object_focus:none"

        # Discovery-beats hook — convergence-threshold scaffolding for the
        # closing arc. Apply BEFORE backdrop / prose so injected REVELATION
        # and BEAT tokens get rendered this turn.
        beats = engine.get("discovery_beats")
        if beats is not None:
            try:
                scene = beats.apply(
                    scene,
                    convergence=list(state.convergence_dims),
                    turn_idx=state.game_turn,
                )
            except Exception as e:
                print(f"[pygame] discovery beats apply failed: {e}", file=sys.stderr)

        # Backdrop swap
        new_loc = scene.get("LOCATION")
        prev_loc = (state.previous_locations[-1]
                    if state.previous_locations else None)
        if new_loc and new_loc != "location:none" and new_loc != prev_loc:
            renderer.set_backdrop(new_loc, crossfade=True)
            renderer.set_location(new_loc.split(":", 1)[-1]
                                   .replace("_", " ").title())

        # Compose prose and render in narration panel
        prose = _structured_compose(engine, scene, state.last_player_card_id)
        renderer.add_clue("CLUE", prose)

        # Convergence tick — minimal placeholder so legacy ending math runs.
        state.convergence_dims = np.minimum(
            1.0, state.convergence_dims + 0.04)
        state.commit_scene(scene)

        # Update journal if available
        if journal is not None:
            try:
                journal.update_from_scene(
                    player_card=state.last_player_card_id,
                    scene=scene,
                    turn=state.game_turn,
                    composed_prose=prose,
                )
            except Exception as e:
                print(f"[pygame] journal update failed: {e}", file=sys.stderr)

        renderer.draw(state.hand, {}, lang, state)
        pygame.time.wait(300)

        if state.game_turn >= spec.max_turns * 2:
            break

    # Build a fake dialogue list for the ending composer.
    fake_dialogue: List[Tuple[str, Token]] = []
    end_lines = _ending_lines(case_data, state.convergence_dims,
                              fake_dialogue, lang, accused, correct_id)
    renderer.set_backdrop(scene_map.get("_ending", None), crossfade=True)
    renderer.show_text_screen("CASE CLOSED" if accused else "THE TRAIL",
                              end_lines, prompt="Press SPACE to exit")
    pygame.quit()


def _structured_wait_for_choice(renderer: "Renderer", n_cards: int,
                                engine: dict, state: StructuredGameState,
                                journal, journal_screen_cls, lang: str) -> str:
    """Variant of wait_for_choice that also handles the J key (journal)."""
    pygame.event.clear()
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return "q"
            if ev.type == pygame.KEYDOWN:
                if pygame.K_1 <= ev.key <= pygame.K_9:
                    n = ev.key - pygame.K_0
                    if 1 <= n <= n_cards:
                        return str(n)
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    return "q"
                if ev.key == pygame.K_a:
                    return "a"
                if ev.key == pygame.K_j:
                    if journal is None or journal_screen_cls is None:
                        # Show a transient notice in the narration panel.
                        renderer.add_transition("Journal not available.")
                        renderer.draw(state.hand, {}, lang, state)
                        continue
                    _open_journal_overlay(renderer, journal,
                                          journal_screen_cls)
                    renderer.draw(state.hand, {}, lang, state)
                    continue
        renderer.clock.tick(FPS)


def _open_journal_overlay(renderer: "Renderer", journal,
                          journal_screen_cls):
    """Open the journal overlay until the user presses J again or Esc."""
    palette = {
        "BG": BG, "INK": INK, "FG": FG, "DIM": DIM, "GOLD": GOLD,
        "SEPIA": SEPIA, "CREAM": CREAM, "PARCHMENT": PARCHMENT,
    }
    fonts = {
        "title": renderer.font_big,
        "header": renderer.font_card,
        "body": renderer.font_serif,
        "italic": renderer.font_serif_i,
        "mono": renderer.font,
        "small": renderer.font_sm,
    }
    try:
        screen_obj = journal_screen_cls(renderer.screen, fonts, palette)
    except Exception as e:
        print(f"[pygame] could not open journal: {e}", file=sys.stderr)
        return
    pygame.event.clear()
    while True:
        try:
            screen_obj.render(journal)
        except Exception as e:
            print(f"[pygame] journal render error: {e}", file=sys.stderr)
            return
        pygame.display.flip()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_j, pygame.K_ESCAPE):
                    return
            try:
                screen_obj.handle_event(ev)
            except Exception:
                pass
        renderer.clock.tick(FPS)


# ─── Briefing / Ending text composition ──────────────────────────────────────
def _briefing_lines(case_data: dict, lang: str) -> List[str]:
    b = case_data.get("briefing", {}).get(lang, case_data.get("briefing", {}).get("en", {}))
    if isinstance(b, str):
        return [b]
    lines = []
    if b.get("setting"):
        lines.append(b["setting"])
        lines.append("")
    if b.get("crime"):
        lines.append(b["crime"])
        lines.append("")
    sus = b.get("suspects", [])
    if sus:
        lines.append("PERSONS OF INTEREST:" if lang == "en"
                     else "PERSONAS DE INTERÉS:" if lang == "es"
                     else "PERSONNES D'INTÉRÊT:")
        for s in sus:
            lines.append(f"  {s.get('name', '')} — {s.get('intro', '')}")
    return lines


def _ending_lines(case_data: dict, conv: np.ndarray, dialogue, lang: str,
                  accused: Optional[Token], correct_id: Optional[str]) -> List[str]:
    score = float(conv.min())
    endings = case_data.get("endings", {})
    fragments = case_data.get("ending_fragments", {})
    lines = []
    if accused and correct_id and accused.id == correct_id:
        if score >= 0.7:
            lines.append("CORRECT. The case was airtight." if lang == "en"
                         else "CORRECTO. El caso fue irrefutable." if lang == "es"
                         else "EXACT. Le dossier était solide.")
            verdict = endings.get("all_strong", {})
        elif score >= 0.4:
            lines.append("CORRECT — but the evidence was thin." if lang == "en"
                         else "CORRECTO — pero la evidencia era débil.")
            verdict = endings.get("lucky_guess", {})
        else:
            lines.append("A lucky guess.")
            verdict = endings.get("lucky_guess", {})
    elif accused:
        correct_name = "the truth"
        lines.append("WRONG accusation." if lang == "en"
                     else "ACUSACIÓN ERRÓNEA." if lang == "es"
                     else "MAUVAISE ACCUSATION.")
        verdict = endings.get("wrong_accusation", {})
    else:
        lines.append("The trail has gone cold." if lang == "en"
                     else "La pista se ha enfriado." if lang == "es"
                     else "La piste s'est refroidie.")
        verdict = endings.get("cold_case", {})

    v = verdict.get(lang, verdict.get("en", ""))
    if v:
        lines.append("")
        lines.append(v)

    # Path-based fragments
    from collections import Counter
    cls_count = Counter()
    for r, t in dialogue:
        if r == "YOU":
            cls_count[t.token_class.value] += 1
    themes = []
    if cls_count.get("OBJECT", 0) + cls_count.get("MODIFIER", 0) >= 3:
        themes.append("physical_evidence")
    if cls_count.get("WITNESS", 0) + cls_count.get("EMOTION", 0) >= 3:
        themes.append("witness_testimony")
    if cls_count.get("MOTIVE", 0) + cls_count.get("ACTION", 0) >= 3:
        themes.append("financial_trail")
    for theme in themes[:2]:
        frag = fragments.get(theme, {}).get(lang, fragments.get(theme, {}).get("en", ""))
        if frag:
            lines.append("")
            lines.append(frag)
    return lines


# ─── Accusation modal ────────────────────────────────────────────────────────
def _accuse(renderer: Renderer, spec: CartridgeSpec) -> Optional[Token]:
    suspects = [t for t in spec.tokens if t.token_class == TokenClass.SUSPECT]
    body = ["Name the culprit. Press the digit of your choice (Esc to cancel)."]
    body.append("")
    for i, s in enumerate(suspects):
        body.append(f"  [{i+1}] {_token_name(s)}")

    renderer.show_text_screen("ACCUSATION", body, prompt="1–9 to accuse, Q to cancel")
    # Wait for digit
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    return None
                if pygame.K_1 <= ev.key <= pygame.K_9:
                    n = ev.key - pygame.K_0 - 1
                    if 0 <= n < len(suspects):
                        return suspects[n]
        renderer.clock.tick(FPS)


# ─── Case select ─────────────────────────────────────────────────────────────
def _scan_cases(trainer_root: Path) -> List[dict]:
    """Return [{id, title, playable, model_path}, ...] for all packed cases."""
    cases_root = trainer_root / "cases"
    out_root = trainer_root / "outputs"
    items = []
    for d in sorted(cases_root.iterdir()):
        if not d.is_dir():
            continue
        spec_p = d / "spec.json"
        if not spec_p.exists():
            continue
        try:
            with open(spec_p) as f:
                spec_j = json.load(f)
            title = spec_j.get("title", d.name)
        except Exception:
            title = d.name
        model_p = out_root / d.name / "dialogue_model.pt"
        items.append({
            "id": d.name,
            "title": title,
            "playable": model_p.exists(),
            "model_path": str(model_p) if model_p.exists() else None,
        })
    return items


def case_select_screen(renderer: Renderer, trainer_root: Path) -> Optional[str]:
    """Show case-select menu (no backdrop). Return case_id or None for quit."""
    cases = _scan_cases(trainer_root)
    if not cases:
        return None

    # First playable as default selection
    sel = next((i for i, c in enumerate(cases) if c["playable"]), 0)
    scroll = 0
    visible_rows = 16

    while True:
        renderer.screen.fill(BG)
        # Title
        t = renderer.font_big.render("LIVING TALES", True, ACCENT)
        renderer.screen.blit(t, (32, 32))
        sub = renderer.font_sm.render(
            "Choose a case  ·  ↑/↓ navigate  ·  Enter to play  ·  Q quit",
            True, DIM)
        renderer.screen.blit(sub, (32, 72))

        # Adjust scroll
        if sel < scroll:
            scroll = sel
        elif sel >= scroll + visible_rows:
            scroll = sel - visible_rows + 1

        y = 120
        for idx in range(scroll, min(scroll + visible_rows, len(cases))):
            c = cases[idx]
            row = pygame.Rect(24, y, W - 48, 36)
            if idx == sel:
                pygame.draw.rect(renderer.screen, (40, 50, 70), row)
                pygame.draw.rect(renderer.screen, ACCENT, row, 1)

            tag_color = CLUE if c["playable"] else (90, 90, 90)
            tag = "●" if c["playable"] else "○"
            tag_s = renderer.font.render(tag, True, tag_color)
            renderer.screen.blit(tag_s, (row.x + 12, row.y + 8))

            title_color = FG if c["playable"] else DIM
            name_s = renderer.font.render(c["title"], True, title_color)
            renderer.screen.blit(name_s, (row.x + 36, row.y + 8))

            id_s = renderer.font_sm.render(c["id"], True, DIM)
            renderer.screen.blit(id_s, (row.x + row.width - id_s.get_width() - 12,
                                        row.y + 10))
            y += 40

        # Footer status
        status = cases[sel]
        if status["playable"]:
            msg = f"✓ Trained model ready  ({Path(status['model_path']).name})"
            mc = CLUE
        else:
            msg = "✗ No trained model — case not playable yet."
            mc = (200, 130, 130)
        renderer.screen.blit(renderer.font_sm.render(msg, True, mc), (32, H - 56))

        if len(cases) > visible_rows:
            page = renderer.font_sm.render(
                f"{sel + 1} / {len(cases)}", True, DIM)
            renderer.screen.blit(page, (W - 80, H - 56))

        pygame.display.flip()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return None
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    return None
                if ev.key in (pygame.K_DOWN, pygame.K_j):
                    sel = (sel + 1) % len(cases)
                elif ev.key in (pygame.K_UP, pygame.K_k):
                    sel = (sel - 1) % len(cases)
                elif ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                    if cases[sel]["playable"]:
                        return cases[sel]["id"]
                    # else: ignore — show "not trained" message stays visible
        renderer.clock.tick(FPS)


# ─── Main loop ───────────────────────────────────────────────────────────────
def run(case_id: str, model_path: Optional[str], lang: str, res: int = 320):
    project_root = _HERE.parent.parent.parent  # 010-more-than-words/
    cases_dir = _HERE.parent / "cases"

    spec, model, mappings = _load_model_and_spec(case_id, model_path)
    if getattr(spec, 'mode', 'converging') == 'oscillating':
        print("[pygame] creature/oscillating mode not supported yet — use TUI.")
        return

    # Language-specific expression overlay
    expr_path = cases_dir / case_id / (
        f"expressions_{lang}.json" if lang != "en" else "expressions.json"
    )
    if not expr_path.exists():
        expr_path = cases_dir / case_id / "expressions.json"
    if expr_path.exists():
        with open(expr_path) as f:
            emap = json.load(f)
        for tok in spec.tokens:
            if tok.id in emap:
                tok.surface_expression = emap[tok.id]

    # Reactions
    reactions = {}
    for p in (cases_dir / case_id / "reactions.json",
              project_root / "cases" / case_id / "reactions.json"):
        if p.exists():
            with open(p) as f:
                reactions = json.load(f)
            break

    # Case data (briefing, endings)
    case_data = {}
    for p in (project_root / "cases" / f"{case_id}.json",
              cases_dir / f"{case_id}.json"):
        if p.exists():
            with open(p) as f:
                case_data = json.load(f)
            break

    # Scene map
    scene_map = {}
    sm_path = cases_dir / case_id / "scene_map.json"
    if sm_path.exists():
        with open(sm_path) as f:
            scene_map = json.load(f)

    correct_id = spec.invariant_token_ids[0] if spec.invariant_token_ids else None

    # Engine version detection: cases shipping `dimensions.json` use the new
    # multidimensional structured engine. Legacy cases keep the old path.
    case_dir = cases_dir / case_id
    if _has_structured_schema(case_dir):
        print(f"[pygame] structured-scene engine for {case_id}")
        _structured_run(case_id, lang, res, project_root, _HERE.parent,
                        spec, scene_map, case_data, correct_id)
        return

    renderer = Renderer(case_id, scene_map, project_root, res=res)
    state = GameState(spec, model, mappings)

    # Title + briefing screens
    renderer.show_text_screen(spec.title, ["A symbolic mystery.", "",
        "Play tokens — the field replies. The truth converges." if lang == "en"
        else "Juega fichas — el campo responde. La verdad converge." if lang == "es"
        else "Jouez des jetons — le terrain répond. La vérité converge."])
    renderer.show_text_screen("", _briefing_lines(case_data, lang))

    # Set the case title using the spec.title (richer than the bare id).
    renderer.case_title = spec.title

    # Open scene
    scene_locs, opening_clues = state.open()
    def _loc_label(tok):
        # Use token id stem ("location:thornfield_crossing" → "Thornfield Crossing"),
        # not surface_expression (which is the long descriptive sentence).
        return tok.id.split(":", 1)[-1].replace("_", " ").title()

    if scene_locs:
        renderer.set_backdrop(scene_locs[0].id, crossfade=True)
        renderer.set_location(_loc_label(scene_locs[0]))
    for tok in scene_locs:
        renderer.add_transition(_get_reaction(tok, None, reactions, lang))
    for tok in opening_clues:
        renderer.add_clue("", _token_narrative(tok))

    accused: Optional[Token] = None
    quit_requested = False
    dim_labels = ["who", "how", "why", "where", "accomplice"]
    progress_signals = {
        "who_advanced": {"en": "The circle of faces tightens.",
                          "es": "El círculo de rostros se estrecha.",
                          "fr": "Le cercle des visages se resserre."},
        "how_advanced": {"en": "The sequence gains a new edge.",
                          "es": "La secuencia gana un borde nuevo.",
                          "fr": "La séquence gagne un nouveau bord."},
        "why_advanced": {"en": "The motive surfaces.",
                          "es": "El móvil emerge.",
                          "fr": "Le mobile émerge."},
        "stagnant":     {"en": "The fog holds.",
                          "es": "La niebla se mantiene.",
                          "fr": "Le brouillard tient."},
        "red_herring":  {"en": "A possibility loses strength.",
                          "es": "Una posibilidad pierde fuerza.",
                          "fr": "Une possibilité perd de sa force."},
    }

    while not state.is_over():
        valid = state.valid_hand()
        # Phase-aware refill if nothing in the current hand is eligible.
        if not valid:
            state.refill_hand()
            valid = state.valid_hand()
        # If still nothing, look ahead in time — bump the turn forward to
        # the next phase that unlocks something. This keeps the game from
        # cold-trail-ending just because the random hand draw missed
        # the current phase window.
        if not valid:
            interlude_lines = {
                "en": "Time passes. The case waits in the fog.",
                "es": "El tiempo pasa. El caso espera en la niebla.",
                "fr": "Le temps passe. L'affaire attend dans la brume.",
            }
            if state.advance_until_eligible():
                state.refill_hand()
                renderer.add_transition(interlude_lines.get(lang, interlude_lines["en"]))
                valid = state.valid_hand()
            if not valid:
                # Genuinely out of evidence — natural ending.
                break

        renderer.draw(valid, reactions, lang, state)
        # Only the first 2 cards are visible (see _draw_hand). Accept the
        # same range — others rotate in as the player consumes the hand.
        choice = renderer.wait_for_choice(min(len(valid), 2))

        if choice == "q":
            quit_requested = True
            break
        if choice == "a":
            picked = _accuse(renderer, spec)
            if picked is not None:
                accused = picked
                break
            continue

        idx = int(choice) - 1
        if idx >= len(valid):
            continue
        tok = valid[idx]

        # Player action — clears prior turn's narration so the panel reads
        # as the current exchange, not a wall of history.
        renderer.reset_narration()
        action_text = reactions.get(tok.id, {}).get("action", _token_narrative(tok))
        renderer.add_clue("YOU", action_text)
        state.play_card(tok)
        renderer.draw(state.valid_hand(), reactions, lang, state)
        pygame.time.wait(300)

        # Engine response
        scene, prev = state.engine_response()
        last_player = state.last_player_token()
        scene_loc = next((t for t in scene if t.token_class == TokenClass.LOCATION), None)
        if scene_loc:
            renderer.set_backdrop(scene_loc.id, crossfade=True)
            renderer.set_location(_loc_label(scene_loc))
            renderer.add_transition(_get_reaction(scene_loc, last_player, reactions, lang))
        clues = [t for t in scene if t.token_class != TokenClass.LOCATION]
        if clues:
            renderer.add_clue("CLUE", _get_reaction(clues[0], last_player, reactions, lang))
            for stok in clues[1:]:
                renderer.add_clue("", _get_reaction(stok, last_player, reactions, lang))

        # Progress signal
        delta = state.convergence_dims - prev
        max_dim = int(delta.argmax()) if delta.max() > 0 else -1
        max_delta = float(delta.max())
        if max_delta < -0.005:
            key = "red_herring"
        elif max_delta < 0.01:
            key = "stagnant"
        elif max_dim == 0:
            key = "who_advanced"
        elif max_dim == 1:
            key = "how_advanced"
        else:
            key = "why_advanced"
        sig = progress_signals[key]
        renderer.set_signal(sig.get(lang, sig.get("en")))

        state.passive_decay()
        renderer.draw(state.valid_hand(), reactions, lang, state)
        pygame.time.wait(400)

    # Ending screen
    if quit_requested and accused is None:
        end_lines = _ending_lines(case_data, state.convergence_dims,
                                  state.dialogue_history, lang, None, correct_id)
    else:
        end_lines = _ending_lines(case_data, state.convergence_dims,
                                  state.dialogue_history, lang, accused, correct_id)

    # Dim final backdrop
    renderer.set_backdrop(scene_map.get("_ending", None), crossfade=True)
    renderer.show_text_screen("CASE CLOSED" if accused else "THE TRAIL",
                              end_lines, prompt="Press SPACE to exit")
    pygame.quit()


# ─── Entry ───────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("case_id", nargs="?", help="Skip case-select if given")
    p.add_argument("--model-path")
    p.add_argument("--lang", default="en")
    p.add_argument("--res", type=int, choices=[240, 320], default=320,
                   help="Backdrop pixel-art source resolution (240 or 320)")
    p.add_argument("--layout", choices=["auto", "full", "compact"], default="auto",
                   help="Window layout: full (H=800, big backdrop) or "
                        "compact (H=720, framed 320 backdrop, fits small laptops)")
    args = p.parse_args()
    if args.layout != "auto":
        os.environ["LIVING_TALES_LAYOUT"] = args.layout

    case_id = args.case_id
    model_path = args.model_path

    if case_id is None:
        # Boot a renderer for the case-select screen (no backdrop yet).
        pygame.init()
        # Force the layout choice now so the menu fits the same window the
        # game will use.
        global W, H, TITLE_RECT, BACKDROP_RECT, LOCATION_RECT
        global NARRATION_RECT, HAND_RECT
        W, H, TITLE_RECT, BACKDROP_RECT, LOCATION_RECT, NARRATION_RECT, HAND_RECT = _rects()
        pygame.display.set_caption("Living Tales")
        screen = pygame.display.set_mode((W, H))
        # Stub renderer just for menu drawing
        class _MenuRenderer:
            def __init__(self):
                self.screen = screen
                self.clock = pygame.time.Clock()
                self.font_big = pygame.font.SysFont(
                    "menlo,monaco,courier", 28, bold=True)
                self.font = pygame.font.SysFont("menlo,monaco,courier", 16)
                self.font_sm = pygame.font.SysFont("menlo,monaco,courier", 13)
        menu = _MenuRenderer()
        case_id = case_select_screen(menu, _HERE.parent)
        pygame.quit()
        if case_id is None:
            print("[pygame] no case selected.")
            return

    if model_path is None:
        default = _HERE.parent / "outputs" / case_id / "dialogue_model.pt"
        if default.exists():
            model_path = str(default)
            print(f"[pygame] using model: {model_path}")

    run(case_id, model_path, args.lang, res=args.res)


if __name__ == "__main__":
    main()
