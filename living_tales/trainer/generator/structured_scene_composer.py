"""
Structured Scene Composer
=========================
Takes a multidimensional scene tuple emitted by `StructuredSceneTransformer`
(or, during Phase 2 development, hand-authored tuples) and assembles an
Obra-Dinn-style third-person past-tense paragraph by slot-filling phrases
authored in `cases/<case>/phrases.json`.

No prose generation. No LLM. The composer is a deterministic templated
grammar that picks the right template for the tuple it receives, fills
the slots, and joins sentences with appropriate punctuation.

Usage
-----
    composer = SceneComposer.load("amber_cipher", lang="en")
    tuple_ = {
        "LOCATION":     "location:station_office",
        "TRANSITION":   "transition:crossed_to",
        "CAUSE":        "cause:following_witness_lead",
        "PRESENCE":     "presence:with_renard_voss",
        "STANCE":       "stance:defensive",
        "ACTION":       "action:examines",
        "OBJECT_FOCUS": "object:initialed_cufflink",
        "TELL":         "tell:tightened",
        "ATMOSPHERE":   "atmosphere:fog_thickens",
        "REVELATION":   "revelation:name_uncovered",
        "BEAT":         "beat:closing_in",
    }
    print(composer.compose(tuple_))
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

_HERE = Path(__file__).resolve().parent
_CASES_DIR = _HERE.parent / "cases"


class SceneComposer:
    """Slot-fills phrase fragments into Obra-Dinn-style scene paragraphs."""

    # Universal NPC-subject suffixes for tell-line composition. The tell phrase
    # often already contains a subject ("his shoulders tightened"); we use this
    # only as a fallback when constructing standalone tell sentences.
    SUBJECT_PRONOUN = {"en": "He", "es": "Él"}
    DETECTIVE_NOUN = {"en": "the detective", "es": "la detective"}

    # Voice-arc stages used for top-NPC phrase variant selection. Determined
    # by per-NPC interview count carried in game_state.
    VOICE_STAGES = ("cold", "warming", "breaking", "broken")

    def __init__(self, case_id: str, phrases: Dict, lang: str = "en"):
        self.case_id = case_id
        self.phrases = phrases
        self.lang = lang
        self._turn_idx: int = 0
        self._npc_interviews: Dict[str, int] = {}

    def set_context(self, turn_idx: int = 0, npc_interview_counts: Optional[Dict[str, int]] = None) -> None:
        """Optional per-call context for variant selection. Composer is a
        deterministic templated grammar — context only affects which *variant*
        of an authored phrase is chosen, never the structure of the prose."""
        self._turn_idx = int(turn_idx)
        self._npc_interviews = dict(npc_interview_counts or {})

    @staticmethod
    def _stage_for_count(n: int) -> str:
        if n <= 0:
            return "cold"
        if n == 1:
            return "warming"
        if n == 2:
            return "breaking"
        return "broken"

    @staticmethod
    def _hash_pick(seq: List[Any], salt: str) -> Any:
        if not seq:
            return None
        h = int(hashlib.sha1(salt.encode("utf-8")).hexdigest(), 16)
        return seq[h % len(seq)]

    # ─── Loading ──────────────────────────────────────────────────────────────
    @classmethod
    def load(cls, case_id: str, lang: str = "en") -> "SceneComposer":
        path = _CASES_DIR / case_id / "phrases.json"
        if not path.exists():
            raise FileNotFoundError(
                f"phrases.json not found for case {case_id} at {path}")
        with open(path) as f:
            phrases = json.load(f)
        return cls(case_id, phrases, lang=lang)

    # ─── Slot lookup ──────────────────────────────────────────────────────────
    def _phrase(self, dim: str, token_id: Optional[str]) -> str:
        """Return the language-specific phrase for a dim/token, or empty string.

        Supported phrases.json entry shapes (all backward-compatible):

        1. Legacy single string per language:
               { "en": "stepped onto", "es": "pisó" }

        2. List of variants per language:
               { "en": ["stepped onto", "moved onto", "set foot on"],
                 "es": [...] }
           Picked deterministically by hash(turn_idx, token_id) so a given
           turn always renders the same variant on replay.

        3. Voice-arc staged variants (top-NPC phrases):
               { "en": { "cold": [...], "warming": [...],
                          "breaking": [...], "broken": [...] },
                 "es": { ... } }
           Stage chosen by interview count for that NPC in game_state.
        """
        if not token_id:
            return ""
        dim_block = self.phrases.get(dim, {})
        entry = dim_block.get(token_id)
        if not entry:
            return ""
        lang_entry = entry.get(self.lang) or entry.get("en")
        if lang_entry is None:
            return ""

        # Shape 1: plain string.
        if isinstance(lang_entry, str):
            return lang_entry

        # Shape 3: dict of stage -> variants.
        if isinstance(lang_entry, dict):
            # Pick stage from per-NPC interview count if this is a PRESENCE-related dim;
            # otherwise default to "cold". The token id encodes the NPC identity for
            # PRESENCE/STANCE/TELL/MEDICAL_TELL/ART_TELL — for now we use the token id
            # itself as the NPC key (callers can re-key via npc_interview_counts).
            count = self._npc_interviews.get(token_id, 0)
            stage = self._stage_for_count(count)
            variants = lang_entry.get(stage) or lang_entry.get("cold")
            if isinstance(variants, str):
                return variants
            if isinstance(variants, list):
                return self._hash_pick(variants, salt=f"{self._turn_idx}:{token_id}:{stage}") or ""
            return ""

        # Shape 2: list of variants.
        if isinstance(lang_entry, list):
            return self._hash_pick(lang_entry, salt=f"{self._turn_idx}:{token_id}") or ""

        return ""

    # ─── Composition ──────────────────────────────────────────────────────────
    def compose(self, tup: Dict[str, str]) -> str:
        """Compose a third-person past-tense scene paragraph from a tuple.

        Sentences are added conditionally — empty phrases are skipped so a
        sparse tuple yields a shorter scene rather than an awkward gap.
        """
        sentences = []

        # Sentence 1: motion — TRANSITION + LOCATION
        s1 = self._sentence_motion(tup)
        if s1:
            sentences.append(s1)

        # Sentence 2: presence + stance
        s2 = self._sentence_presence(tup)
        if s2:
            sentences.append(s2)

        # Sentence 3: detective's action + object
        s3 = self._sentence_action(tup)
        if s3:
            sentences.append(s3)

        # Sentence 4: tell (only if NPC present and tell != none)
        s4 = self._sentence_tell(tup)
        if s4:
            sentences.append(s4)

        # Sentence 5: atmosphere
        atm = self._phrase("ATMOSPHERE", tup.get("ATMOSPHERE"))
        if atm:
            sentences.append(atm)

        # Sentence 6: revelation
        rev = self._phrase("REVELATION", tup.get("REVELATION"))
        if rev:
            sentences.append(rev)

        # Optional: beat-flagged closing line (only `closing_in` and
        # `verdict_ready` carry visible prose)
        beat = self._phrase("BEAT", tup.get("BEAT"))
        if beat:
            sentences.append(beat)

        return " ".join(self._terminate(s) for s in sentences if s)

    # ─── Sentence builders ────────────────────────────────────────────────────
    def _sentence_motion(self, tup: Dict[str, str]) -> str:
        cause = self._phrase("CAUSE", tup.get("CAUSE"))
        transition = self._phrase("TRANSITION", tup.get("TRANSITION"))
        location = self._phrase("LOCATION", tup.get("LOCATION"))
        if not (transition or location):
            return ""
        # If TRANSITION is "Began at" / "Comenzó en", LOCATION is the place
        # the case opens at — no cause prefix.
        opener = " ".join(p for p in [transition, location] if p).strip()
        if cause:
            # "Following the witness's word, crossed to the station office"
            opener = f"{cause} {opener[0].lower()}{opener[1:]}" if opener else cause
        return opener

    def _sentence_presence(self, tup: Dict[str, str]) -> str:
        presence = self._phrase("PRESENCE", tup.get("PRESENCE"))
        stance = self._phrase("STANCE", tup.get("STANCE"))
        if not presence:
            return ""
        # Capitalise the first letter — phrases are authored lowercase to
        # support mid-sentence inlining, but at sentence head they need caps.
        presence = self._sentence_caps(presence)
        if stance:
            return f"{presence}, {stance}"
        return presence

    # Verbs that take an OBJECT_FOCUS as the grammatical object.
    _ACTION_TAKES_OBJECT = {"action:examines"}
    # Verbs of motion — they take a LOCATION as the destination, not an
    # OBJECT_FOCUS. Pairing with OBJECT_FOCUS produces "stepped into the
    # satchel" / "stepped into the cufflink" — nonsense. We drop the
    # sentence (motion is already covered by the TRANSITION line) when no
    # location is available.
    _ACTION_TAKES_LOCATION = {"action:arrives", "action:leaves"}

    def _sentence_action(self, tup: Dict[str, str]) -> str:
        action_id = tup.get("ACTION")
        action = self._phrase("ACTION", action_id)
        if not action or action_id == "action:none":
            return ""
        # Motion verbs: skip if no location, else pair with LOCATION.
        if action_id in self._ACTION_TAKES_LOCATION:
            loc = self._phrase("LOCATION", tup.get("LOCATION"))
            if not loc:
                return ""
            subject = self.DETECTIVE_NOUN.get(self.lang, self.DETECTIVE_NOUN["en"])
            return f"{subject.capitalize()} {action} {loc}"
        # Object-taking verbs: skip if no object.
        obj = self._phrase("OBJECT_FOCUS", tup.get("OBJECT_FOCUS"))
        if not obj and action_id in self._ACTION_TAKES_OBJECT:
            return ""
        subject = self.DETECTIVE_NOUN.get(self.lang, self.DETECTIVE_NOUN["en"])
        if obj:
            return f"{subject.capitalize()} {action} {obj}"
        return f"{subject.capitalize()} {action}"

    def _sentence_tell(self, tup: Dict[str, str]) -> str:
        if tup.get("PRESENCE") == "presence:alone":
            return ""
        tell = self._phrase("TELL", tup.get("TELL"))
        if not tell:
            return ""
        return self._sentence_caps(tell)

    @staticmethod
    def _sentence_caps(s: str) -> str:
        if not s:
            return s
        return s[0].upper() + s[1:]

    # ─── Punctuation + language fixups ────────────────────────────────────────
    # Spanish contractions that the slot-fill produces in awkward forms.
    _ES_FIXUPS = [
        (" a el ", " al "),
        (" de el ", " del "),
    ]

    @classmethod
    def _terminate(cls, sentence: str) -> str:
        s = sentence.strip()
        if not s:
            return ""
        # Apply language fixups (cheap, safe; only catches the produced patterns)
        for pat, rep in cls._ES_FIXUPS:
            s = s.replace(pat, rep)
        if s[-1] in ".!?…":
            return s
        return s + "."


# ─────────────────────────────────────────────────────────────────────────────
# Self-test — runs the composer against synthetic tuples to verify the
# Phase 3 deliverable: prose comes out reading as Obra-Dinn-style scenes
# without any model retraining.
# ─────────────────────────────────────────────────────────────────────────────
def _selftest():
    composer_en = SceneComposer.load("amber_cipher", lang="en")
    composer_es = SceneComposer.load("amber_cipher", lang="es")

    test_tuples = [
        {
            "_label": "Opening — alone at Thornfield",
            "LOCATION":     "location:thornfield_crossing",
            "TRANSITION":   "transition:none",
            "CAUSE":        "cause:none",
            "PRESENCE":     "presence:alone",
            "STANCE":       "stance:none",
            "ACTION":       "action:arrives",
            "OBJECT_FOCUS": "object_focus:none",
            "TELL":         "tell:none",
            "ATMOSPHERE":   "atmosphere:fog_thickens",
            "REVELATION":   "revelation:none",
            "BEAT":         "beat:orientation",
        },
        {
            "_label": "Cufflink revelation with Voss in office",
            "LOCATION":     "location:station_office",
            "TRANSITION":   "transition:crossed_to",
            "CAUSE":        "cause:following_witness_lead",
            "PRESENCE":     "presence:with_renard_voss",
            "STANCE":       "stance:defensive",
            "ACTION":       "action:examines",
            "OBJECT_FOCUS": "object:initialed_cufflink",
            "TELL":         "tell:tightened",
            "ATMOSPHERE":   "atmosphere:fog_thickens",
            "REVELATION":   "revelation:name_uncovered",
            "BEAT":         "beat:closing_in",
        },
        {
            "_label": "Mid-game witness interrogation",
            "LOCATION":     "location:platform_two",
            "TRANSITION":   "transition:stayed",
            "CAUSE":        "cause:noticed_inconsistency",
            "PRESENCE":     "presence:with_signalman",
            "STANCE":       "stance:evasive",
            "ACTION":       "action:questions",
            "OBJECT_FOCUS": "object_focus:none",
            "TELL":         "emotion:paused_reply",
            "ATMOSPHERE":   "atmosphere:silence_holds",
            "REVELATION":   "revelation:contradiction_surfaces",
            "BEAT":         "beat:investigation",
        },
        {
            "_label": "Late-game verdict beat",
            "LOCATION":     "location:goods_shed",
            "TRANSITION":   "transition:descended_to",
            "CAUSE":        "cause:pursued_suspicion",
            "PRESENCE":     "presence:with_renard_voss",
            "STANCE":       "stance:hostile",
            "ACTION":       "action:confronts",
            "OBJECT_FOCUS": "object:cipher_sheet",
            "TELL":         "tell:paled",
            "ATMOSPHERE":   "atmosphere:dawn_approaches",
            "REVELATION":   "revelation:motive_emerges",
            "BEAT":         "beat:verdict_ready",
        },
    ]

    for t in test_tuples:
        label = t.pop("_label")
        print(f"━━━ {label} ━━━")
        print(f"[en] {composer_en.compose(t)}")
        print(f"[es] {composer_es.compose(t)}")
        print()


if __name__ == "__main__":
    _selftest()
