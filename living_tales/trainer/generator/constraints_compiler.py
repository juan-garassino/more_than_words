"""
Constraints Compiler
====================
Compiles `cases/<case>/constraints.json` hard-mask rules into a runtime
mask + validator. Two entry points:

- `applicable_for_dim(dim, scene_so_far, game_state)` returns the set of
  legal token IDs for that dim given dims emitted earlier in the scene
  and the current game state. Used at inference to mask logits.

- `is_valid_tuple(scene, game_state)` — full-scene check used by the
  trajectory validator to flag authoring mistakes. Returns
  (ok: bool, violated_rule_ids: list[str]).

Game state shape (dict)
-----------------------
    {
      "previous_locations": list[str],   # LOCATION ids of prior scenes,
                                         #   most recent last (excl. `none`)
      "visited_locations":  set[str],    # all LOCATION ids ever visited
      "scene_index":        int,         # 0-indexed; 0 == first scene
      "convergence_dims":   list[float], # current convergence per attractor
      "game_turn":          int,         # 1-indexed turn number
      "last_player_card":   str | None,  # token id of player's most recent card
    }
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _is_alone(presence_value: Optional[str]) -> bool:
    return presence_value == "presence:alone"


def _is_witness_or_suspect(presence_value: Optional[str]) -> bool:
    if not presence_value:
        return False
    if presence_value == "presence:alone":
        return False
    return presence_value.startswith("presence:with_")


def _convergence_min(dims: Optional[List[float]]) -> float:
    if not dims:
        return 0.0
    return min(dims)


# ─── Mask compiler / validator ───────────────────────────────────────────────
class ConstraintMask:
    """Runtime hard-mask compiler and validator."""

    def __init__(
        self,
        constraints_json: Dict[str, Any],
        dim_vocab: Dict[str, List[str]],
        token_class_map: Optional[Dict[str, str]] = None,
    ):
        self.rules: List[Dict[str, Any]] = list(
            constraints_json.get("rules", [])
        )
        self.dim_vocab: Dict[str, List[str]] = {
            k: list(v) for k, v in dim_vocab.items()
        }
        # Map token_id -> token_class (e.g. "WITNESS", "SUSPECT", "OBJECT").
        self.token_class_map: Dict[str, str] = dict(token_class_map or {})

    # ─── Inference-time masking ──────────────────────────────────────────
    def applicable_for_dim(
        self,
        dim: str,
        scene_so_far: Dict[str, str],
        game_state: Dict[str, Any],
    ) -> Set[str]:
        """Return the set of legal token IDs for `dim` given partial scene
        + state.

        Two passes per rule:

        Forward — rule's `if` matches scene_so_far, and rule's `then`
        constrains `dim`. Intersect dim's vocab with the allowed set.

        Reverse — rule's `if` is keyed on `dim` (with a literal token
        value) and the `then` is over already-emitted dims / state.
        For each candidate value in dim, hypothetically set dim=candidate
        and evaluate `then`; if `then` would be violated, drop the
        candidate. This catches sequential-emission cases where a later
        dim's value is constrained by earlier ones — e.g.
        `if TRANSITION:returned then LOCATION:@in_visited_locations`
        must filter `transition:returned` out at TRANSITION-emission time
        when LOCATION is already chosen but not in visited.
        """
        vocab: Set[str] = set(self.dim_vocab.get(dim, []))
        allowed: Set[str] = set(vocab)
        for rule in self.rules:
            cond = rule.get("if", {})
            then = rule.get("then", {})

            # Forward direction.
            if self._cond_matches(cond, scene_so_far, game_state):
                allowed &= self._then_allowed_for_dim(
                    dim, then, scene_so_far, game_state, rule
                )

            # Reverse direction: rule keyed on this dim with a literal
            # value, then over prior dims / state.
            if (
                list(cond.keys()) == [dim]
                and isinstance(cond.get(dim), str)
                and not cond[dim].startswith("@")
            ):
                candidate = cond[dim]
                if candidate in allowed:
                    hypo = dict(scene_so_far)
                    hypo[dim] = candidate
                    # Only enforce if the `then` is over dims/state already
                    # determined — checking `then` references only keys in
                    # `hypo` or are state-level (start with @).
                    can_evaluate = all(
                        k.startswith("@") or k == "values" or k in hypo
                        for k in then.keys()
                    )
                    if can_evaluate and not self._then_holds(
                        then, hypo, game_state, rule
                    ):
                        allowed.discard(candidate)

        return allowed

    # ─── Validator ───────────────────────────────────────────────────────
    def is_valid_tuple(
        self,
        scene: Dict[str, str],
        game_state: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        violated: List[str] = []
        for rule in self.rules:
            cond = rule.get("if", {})
            then = rule.get("then", {})
            if not self._cond_matches(cond, scene, game_state):
                continue
            if not self._then_holds(then, scene, game_state, rule):
                violated.append(rule.get("id", "<unnamed_rule>"))
        return (len(violated) == 0, violated)

    # ─── Condition evaluation ────────────────────────────────────────────
    def _cond_matches(
        self,
        cond: Dict[str, Any],
        scene: Dict[str, str],
        state: Dict[str, Any],
    ) -> bool:
        """All clauses in `cond` must hold for the rule to fire."""
        for key, val in cond.items():
            if not self._clause_holds(key, val, scene, state, cond):
                return False
        return True

    def _clause_holds(
        self,
        key: str,
        val: Any,
        scene: Dict[str, str],
        state: Dict[str, Any],
        sibling: Dict[str, Any],
    ) -> bool:
        """Evaluate a single key:value clause from an `if` block."""
        # Predicate keys (start with @) are state-level checks.
        if key.startswith("@"):
            return self._eval_predicate(key, val, scene, state, sibling, None)
        # Otherwise a dim with a literal token id or a predicate value.
        actual = scene.get(key)
        if isinstance(val, str) and val.startswith("@"):
            return self._eval_dim_predicate(
                key, val, sibling, actual, scene, state
            )
        return actual == val

    # ─── `then` evaluation ───────────────────────────────────────────────
    def _then_holds(
        self,
        then: Dict[str, Any],
        scene: Dict[str, str],
        state: Dict[str, Any],
        rule: Dict[str, Any],
    ) -> bool:
        for key, val in then.items():
            if key == "values":
                continue  # consumed alongside the predicate sibling
            if key.startswith("@"):
                if not self._eval_predicate(key, val, scene, state, then, rule):
                    return False
                continue
            actual = scene.get(key)
            if isinstance(val, str) and val.startswith("@"):
                if not self._eval_dim_predicate(
                    key, val, then, actual, scene, state
                ):
                    return False
            else:
                if actual != val:
                    return False
        return True

    def _then_allowed_for_dim(
        self,
        dim: str,
        then: Dict[str, Any],
        scene: Dict[str, str],
        state: Dict[str, Any],
        rule: Dict[str, Any],
    ) -> Set[str]:
        """Reduce `then` to the set of tokens still legal for `dim`."""
        vocab = set(self.dim_vocab.get(dim, []))
        # State-only predicates in `then` (e.g. @convergence_min_gte) don't
        # restrict any dim's vocab — they either hold or they don't, and
        # if they don't hold, the rule's `if` should not have fired in the
        # first place at sample time. For mask compilation, treat them as
        # non-binding on dim vocab.
        if dim not in then:
            return vocab
        val = then[dim]
        if isinstance(val, str) and not val.startswith("@"):
            return {val} & vocab
        if isinstance(val, str) and val.startswith("@"):
            return self._predicate_allowed_set(
                dim, val, then, scene, state, vocab
            )
        return vocab

    # ─── Dim-predicate handlers (value-level) ────────────────────────────
    def _eval_dim_predicate(
        self,
        dim: str,
        pred: str,
        sibling: Dict[str, Any],
        actual: Optional[str],
        scene: Dict[str, str],
        state: Dict[str, Any],
    ) -> bool:
        allowed = self._predicate_allowed_set(
            dim, pred, sibling, scene, state,
            set(self.dim_vocab.get(dim, [])),
        )
        return actual in allowed

    def _predicate_allowed_set(
        self,
        dim: str,
        pred: str,
        sibling: Dict[str, Any],
        scene: Dict[str, str],
        state: Dict[str, Any],
        vocab: Set[str],
    ) -> Set[str]:
        """Return the subset of `vocab` allowed by predicate `pred`."""
        if pred == "@equals_previous_scene_location":
            prev = state.get("previous_locations") or []
            if not prev:
                return set()
            return {prev[-1]} & vocab
        if pred == "@differs_from_previous_scene_location":
            prev = state.get("previous_locations") or []
            last = prev[-1] if prev else None
            return {
                t for t in vocab
                if t != last and t != "location:none"
            }
        if pred == "@in_visited_locations":
            visited = state.get("visited_locations") or set()
            return set(visited) & vocab
        if pred == "@not_alone":
            return {t for t in vocab if t != "presence:alone"}
        if pred == "@is_witness_or_suspect":
            return {t for t in vocab if _is_witness_or_suspect(t)}
        if pred == "@in":
            values = sibling.get("values", [])
            return set(values) & vocab
        if pred == "@not_in":
            values = set(sibling.get("values", []))
            return {t for t in vocab if t not in values}
        # Unknown predicate: allow everything, validator will catch via
        # _eval_predicate fallback if it's actually state-based.
        return vocab

    # ─── State predicate handlers ────────────────────────────────────────
    def _eval_predicate(
        self,
        pred: str,
        val: Any,
        scene: Dict[str, str],
        state: Dict[str, Any],
        sibling: Dict[str, Any],
        rule: Optional[Dict[str, Any]],
    ) -> bool:
        if pred == "@scene_index_eq":
            return int(state.get("scene_index", -1)) == int(val)
        if pred == "@convergence_min_gte":
            return _convergence_min(state.get("convergence_dims")) >= float(val)
        if pred == "@convergence_min_lt":
            return _convergence_min(state.get("convergence_dims")) < float(val)
        if pred == "@turn_gte":
            return int(state.get("game_turn", 0)) >= int(val)
        if pred == "@not_alone":
            return not _is_alone(scene.get("PRESENCE"))
        if pred == "@is_witness_or_suspect":
            return _is_witness_or_suspect(scene.get("PRESENCE"))
        if pred == "@last_player_token_class_in":
            classes = set(val) if isinstance(val, list) else {val}
            last_card = state.get("last_player_card")
            if not last_card:
                return False
            cls = self.token_class_map.get(last_card)
            if cls is None:
                # Heuristic: derive class from token id prefix when the
                # mapping is incomplete.
                prefix = last_card.split(":", 1)[0].upper()
                cls = {
                    "WITNESS": "WITNESS",
                    "SUSPECT": "SUSPECT",
                    "OBJECT": "OBJECT",
                    "MODIFIER": "MODIFIER",
                    "MOTIVE": "MOTIVE",
                    "EVENT": "EVENT",
                    "TIME": "TIME",
                    "ACTION": "ACTION",
                    "TRAVEL": "TRAVEL",
                }.get(prefix, prefix)
            return cls in classes
        # Unknown predicate — be permissive so we don't false-fail.
        return True
