#!/usr/bin/env python3
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Original amber_cipher spec (3-dim, 72 tokens) — used when n_dims==3
CLASSES_SPEC_3 = {
    'SUSPECT':7,
    'LOCATION':8,
    'OBJECT':13,
    'EMOTION':6,
    'MODIFIER':10,
    'ACTION':9,
    'TIME':5,
    'MOTIVE':6,
    'WITNESS':5,
    'EVENT':3,
}
PHASE_COUNTS_3 = {'EARLY':20,'MID':33,'LATE':16,'INVARIANT':3}

RED_TAGS = {"surface","plausible","visible","dramatic"}
KILLER_TAGS = {"access","error","guilt","concealment"}
MECH_TAGS = {"physical","time_window","method","constraint"}
MOTIVE_TAGS = {"buried","financial","prior","secret"}

SENTENCE_SPLIT = re.compile(r"[.!?]+")


def sentence_count(text: str) -> int:
    # Count non-empty sentence segments, ignore abbreviations like R.V. by removing single-letter initials
    cleaned = re.sub(r"\b([A-Z])\.\b", r"\1", text)
    parts = [p.strip() for p in SENTENCE_SPLIT.split(cleaned) if p.strip()]
    return len(parts)


def load_case(path: Path):
    return json.loads(path.read_text())


def check(path: Path):
    case = load_case(path)
    violations = []

    tokens = case.get('tokens', [])
    token_by_id = {t['id']: t for t in tokens}

    # Determine n_dims from the attractor section
    attractor_dims = case.get('attractor', {}).get('dimensions', [])
    n_dims = len(attractor_dims) if attractor_dims else 3

    # Determine convergence params from case JSON or defaults
    convergence_rate = case.get('convergence_rate', 0.40)
    max_turns = case.get('max_turns', 18)

    def fail(name, msg):
        violations.append((name, msg))

    # STRUCTURE
    inv = [t for t in tokens if t.get('is_invariant')]
    expected_invariants = len(case.get('invariant_token_ids', []))

    name = 'STRUCTURE invariants count'
    if len(inv) == expected_invariants:
        pass
    else:
        fail(name, f"expected {expected_invariants} invariants (from invariant_token_ids), got {len(inv)}")

    name = 'STRUCTURE tokens'
    n_tokens = len(tokens)
    if n_dims == 3:
        if n_tokens == 72:
            pass
        else:
            fail(name, f"expected 72 tokens (3-dim case), got {n_tokens}")
    else:
        # For larger cases, just verify token count is positive and consistent
        if n_tokens < 72:
            fail(name, f"expected at least 72 tokens for {n_dims}-dim case, got {n_tokens}")

    if n_dims == 3:
        name = 'STRUCTURE class distribution'
        counts = Counter(t['class'] for t in tokens if not t.get('is_invariant'))
        # Compare only non-invariant tokens to CLASSES_SPEC_3 (which excludes invariants)
        expected_non_inv = {k: v for k, v in CLASSES_SPEC_3.items()}
        # Note: original spec includes the 3 invariants (1 SUSPECT, 1 EVENT, 1 MOTIVE)
        # so we adjust: full token counts including invariants were in CLASSES_SPEC_3
        full_counts = Counter(t['class'] for t in tokens)
        if full_counts == CLASSES_SPEC_3:
            pass
        else:
            fail(name, f"expected {CLASSES_SPEC_3}, got {dict(full_counts)}")

        name = 'PHASES counts'
        phase_counts = Counter(t['phase'] for t in tokens)
        if phase_counts == PHASE_COUNTS_3:
            pass
        else:
            fail(name, f"expected {PHASE_COUNTS_3}, got {dict(phase_counts)}")
    else:
        # For larger cases: just check that all phases are valid values
        name = 'PHASES valid values'
        valid_phases = {'EARLY', 'MID', 'LATE', 'INVARIANT'}
        bad_phases = [t['id'] for t in tokens if t['phase'] not in valid_phases]
        if bad_phases:
            fail(name, f"invalid phase values: {bad_phases[:10]}")

        name = 'PHASES invariant phase for invariant tokens'
        bad_phase = [t['id'] for t in inv if t['phase'] != 'INVARIANT']
        if bad_phase:
            fail(name, f"invariant tokens with non-INVARIANT phase: {bad_phase}")

    name = 'PHASES victim token early'
    victim = next((t for t in tokens if t['id'].startswith('event:') and t['id'].endswith('_discovered')), None)
    if victim and victim['phase'] == 'EARLY':
        pass
    else:
        fail(name, f"victim token not EARLY: {victim['id'] if victim else 'missing'}")

    name = 'PHASES killer suspect not EARLY'
    killer = next((t for t in tokens if t['class']=='SUSPECT' and t['is_invariant']), None)
    if killer and killer['phase'] != 'EARLY':
        pass
    else:
        fail(name, f"killer suspect EARLY or missing: {killer['id'] if killer else 'missing'}")

    name = 'PHASES invariants INVARIANT'
    bad_phase = [t['id'] for t in inv if t['phase'] != 'INVARIANT']
    if bad_phase:
        fail(name, f"invariant phase not INVARIANT: {bad_phase}")

    # ATTRACTOR WEIGHTS — validate vector length matches n_dims
    name = 'WEIGHTS vector length'
    bad_len = [(t['id'], len(t['attractor_weights'])) for t in tokens if len(t['attractor_weights']) != n_dims]
    if bad_len:
        fail(name, f"wrong weight vector length (expected {n_dims}): {bad_len[:5]}")

    name = 'WEIGHTS values in [0,1]'
    out_of_range = []
    for t in tokens:
        w = t['attractor_weights']
        if any(v < 0.0 or v > 1.0 for v in w):
            out_of_range.append((t['id'], w))
    if out_of_range:
        fail(name, f"out of range: {out_of_range[:5]}")

    name = 'STRUCTURE invariant unit vectors'
    bad = []
    for t in inv:
        w = t['attractor_weights']
        # Unit vector: exactly one 1.0, rest 0.0
        ones = [i for i, v in enumerate(w) if v == 1.0]
        zeros = [i for i, v in enumerate(w) if v == 0.0]
        if len(ones) != 1 or len(zeros) != n_dims - 1:
            bad.append((t['id'], w))
    if bad:
        fail(name, f"bad invariant vectors: {bad}")

    name = 'STRUCTURE token id format'
    bad_ids = []
    for t in tokens:
        tid = t['id']
        if ' ' in tid:
            bad_ids.append(tid)
        part = tid.split(':',1)[1] if ':' in tid else tid
        if 'the' in part:
            bad_ids.append(tid)
    if bad_ids:
        fail(name, f"bad ids: {bad_ids[:10]}")

    def sumw(t):
        return sum(t['attractor_weights'])

    name = 'WEIGHTS early range'
    bad = [t['id'] for t in tokens if t['phase']=='EARLY' and not (0.08 <= sumw(t) <= 0.28 * n_dims)]
    if bad:
        fail(name, f"early outside range: {bad[:10]}")

    name = 'WEIGHTS mid range'
    bad = [t['id'] for t in tokens if t['phase']=='MID' and not (0.22 <= sumw(t) <= 0.52 * n_dims)]
    if bad:
        fail(name, f"mid outside range: {bad[:10]}")

    name = 'WEIGHTS late range'
    bad = [t['id'] for t in tokens if t['phase']=='LATE' and not (0.45 <= sumw(t) <= 0.80 * n_dims)]
    if bad:
        fail(name, f"late outside range: {bad[:10]}")

    name = 'WEIGHTS red herring max'
    bad = [t['id'] for t in tokens if RED_TAGS.issubset(set(t['affinity_tags'])) and sumw(t) > 0.38 * n_dims]
    if bad:
        fail(name, f"red herring too high: {bad[:10]}")

    # GRAPH
    name = 'GRAPH self-loops'
    edges = case.get('graph', {}).get('edges', [])
    loops = [e for e in edges if e['from'] == e['to']]
    if loops:
        fail(name, f"self-loops: {loops[:3]}")

    name = 'GRAPH duplicate/symmetric'
    seen = set()
    dup = []
    for e in edges:
        key = tuple(sorted([e['from'], e['to']]))
        if key in seen:
            dup.append(key)
        seen.add(key)
    if dup:
        fail(name, f"duplicate edges: {dup[:5]}")

    # Degree bounds: scale with vocab size
    max_degree = max(8, n_tokens // 5)
    name = 'GRAPH degree bounds'
    degree = Counter()
    for e in edges:
        degree[e['from']] += 1
        degree[e['to']] += 1
    low = [n for n in case['graph']['nodes'] if degree[n] < 2]
    high = [n for n in case['graph']['nodes'] if degree[n] > max_degree]
    if low or high:
        fail(name, f"degree low:{low[:5]} high:{high[:5]}")

    name = 'GRAPH invariant isolation'
    inv_ids = [t['id'] for t in inv]
    inv_set = set(inv_ids)
    bad = []
    for e in edges:
        if e['from'] in inv_set and e['to'] in inv_set:
            bad.append((e['from'], e['to']))
    if bad:
        fail(name, f"invariant edges: {bad[:5]}")

    name = 'GRAPH red tag overlap'
    overlap = []
    for t in tokens:
        tags = set(t['affinity_tags'])
        if tags & RED_TAGS:
            if tags & (KILLER_TAGS | MECH_TAGS | MOTIVE_TAGS):
                overlap.append(t['id'])
    if overlap:
        fail(name, f"red overlap: {overlap[:10]}")

    # EXPRESSIONS
    name = 'EXPRESSIONS coverage'
    expressions = case.get('expressions', {})
    missing = [t['id'] for t in tokens if t['id'] not in expressions]
    if missing:
        fail(name, f"missing expressions: {missing[:10]}")

    name = 'EXPRESSIONS sentence count'
    too_many = []
    for tid, text in expressions.items():
        if sentence_count(text) > 2:
            too_many.append(tid)
    if too_many:
        fail(name, f"too many sentences: {too_many[:10]}")

    name = 'EXPRESSIONS template'
    templ = []
    for tid, text in expressions.items():
        if '<' in text or '>' in text or 'PLACEHOLDER' in text or 'TEMPLATE' in text:
            templ.append(tid)
    if templ:
        fail(name, f"template markers: {templ[:10]}")

    # CONVERGENCE SIMULATION — adaptive for n_dims
    def simulate(order_ids):
        dims = [0.0] * n_dims
        cap = min(len(order_ids), max_turns)
        for i, tid in enumerate(order_ids[:cap]):
            w = token_by_id[tid]['attractor_weights']
            dims = [min(1.0, d + w[j] * convergence_rate) for j, d in enumerate(dims)]
            if min(dims) >= 0.75:
                return True
        return False

    name = 'SIM A top signal descending'
    non_inv = [t for t in tokens if not t['is_invariant']]
    if n_dims == 3:
        # For 3-dim: sort by killer dim weight
        order = sorted(non_inv, key=lambda t: t['attractor_weights'][0], reverse=True)
    else:
        # For n_dims > 3: sort by sum of all weights (ensures coverage across all dims)
        order = sorted(non_inv, key=lambda t: sum(t['attractor_weights']), reverse=True)
    if not simulate([t['id'] for t in order]):
        fail(name, f"did not converge by turn {max_turns}")

    name = 'SIM B locations first'
    locs = [t for t in non_inv if t['class']=='LOCATION']
    rest = [t for t in non_inv if t['class']!='LOCATION']
    order = locs + sorted(rest, key=lambda t: sum(t['attractor_weights']), reverse=True)
    if not simulate([t['id'] for t in order]):
        fail(name, f"did not converge by turn {max_turns}")

    name = 'SIM C red herring then enabler'
    red = [t for t in non_inv if RED_TAGS.issubset(set(t['affinity_tags']))]
    # find enabler: suspect with dim0>=0.2 and dim1>=0.2
    enabler = None
    for t in non_inv:
        if t['class']=='SUSPECT' and t['attractor_weights'][0] >= 0.2 and t['attractor_weights'][1] >= 0.2:
            enabler = t
            break
    rest_c = sorted([t for t in non_inv if t not in red and t != enabler],
                    key=lambda t: sum(t['attractor_weights']), reverse=True)
    order = red + ([enabler] if enabler else []) + rest_c
    if not simulate([t['id'] for t in order]):
        fail(name, f"did not converge by turn {max_turns}")

    return violations


def main(argv):
    if len(argv) < 2:
        print("Usage: thornfield_case_validator.py <case.json> [case.json ...]")
        sys.exit(1)

    total_violations = 0
    for arg in argv[1:]:
        path = Path(arg)
        print(f"== {path.name} ==")
        violations = check(path)
        if not violations:
            print("PASS all checks")
        else:
            for name, msg in violations:
                print(f"FAIL {name}: {msg}")
            total_violations += len(violations)
        print("")

    if total_violations == 0:
        print("OVERALL PASS (0 violations)")
        sys.exit(0)
    else:
        print(f"OVERALL FAIL ({total_violations} violations)")
        sys.exit(2)


if __name__ == '__main__':
    main(sys.argv)
