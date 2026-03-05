#!/usr/bin/env python3
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

CLASSES_SPEC = {
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

PHASE_COUNTS = {'EARLY':20,'MID':33,'LATE':16,'INVARIANT':3}

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

    def fail(name, msg):
        violations.append((name, msg))

    # STRUCTURE
    name = 'STRUCTURE tokens'
    if len(tokens) == 72:
        pass
    else:
        fail(name, f"expected 72 tokens, got {len(tokens)}")

    name = 'STRUCTURE class distribution'
    counts = Counter(t['class'] for t in tokens)
    if counts == CLASSES_SPEC:
        pass
    else:
        fail(name, f"expected {CLASSES_SPEC}, got {dict(counts)}")

    name = 'STRUCTURE invariants count'
    inv = [t for t in tokens if t.get('is_invariant')]
    if len(inv) == 3:
        pass
    else:
        fail(name, f"expected 3 invariants, got {len(inv)}")

    name = 'STRUCTURE invariant unit vectors'
    bad = []
    for t in inv:
        w = t['attractor_weights']
        if t['class'] == 'SUSPECT' and w != [1.0,0.0,0.0]:
            bad.append((t['id'], w))
        if t['class'] == 'EVENT' and w != [0.0,1.0,0.0]:
            bad.append((t['id'], w))
        if t['class'] == 'MOTIVE' and w != [0.0,0.0,1.0]:
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

    # PHASES
    name = 'PHASES counts'
    phase_counts = Counter(t['phase'] for t in tokens)
    if phase_counts == PHASE_COUNTS:
        pass
    else:
        fail(name, f"expected {PHASE_COUNTS}, got {dict(phase_counts)}")

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

    # ATTRACTOR WEIGHTS
    name = 'WEIGHTS values in [0,1]'
    out_of_range = []
    for t in tokens:
        w = t['attractor_weights']
        if any(v < 0.0 or v > 1.0 for v in w):
            out_of_range.append((t['id'], w))
    if out_of_range:
        fail(name, f"out of range: {out_of_range[:5]}")

    def sumw(t):
        return sum(t['attractor_weights'])

    name = 'WEIGHTS early range'
    bad = [t['id'] for t in tokens if t['phase']=='EARLY' and not (0.08 <= sumw(t) <= 0.28)]
    if bad:
        fail(name, f"early outside range: {bad[:10]}")

    name = 'WEIGHTS mid range'
    bad = [t['id'] for t in tokens if t['phase']=='MID' and not (0.22 <= sumw(t) <= 0.52)]
    if bad:
        fail(name, f"mid outside range: {bad[:10]}")

    name = 'WEIGHTS late range'
    bad = [t['id'] for t in tokens if t['phase']=='LATE' and not (0.45 <= sumw(t) <= 0.80)]
    if bad:
        fail(name, f"late outside range: {bad[:10]}")

    name = 'WEIGHTS red herring max'
    bad = [t['id'] for t in tokens if RED_TAGS.issubset(set(t['affinity_tags'])) and sumw(t) > 0.38]
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

    name = 'GRAPH degree bounds'
    degree = Counter()
    for e in edges:
        degree[e['from']] += 1
        degree[e['to']] += 1
    low = [n for n in case['graph']['nodes'] if degree[n] < 2]
    high = [n for n in case['graph']['nodes'] if degree[n] > 8]
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

    # CONVERGENCE SIMULATION
    def simulate(order_ids):
        dims = [0.0,0.0,0.0]
        for i, tid in enumerate(order_ids[:18]):
            w = token_by_id[tid]['attractor_weights']
            dims = [min(1.0, d + w[j]*0.25) for j,d in enumerate(dims)]
            if min(dims) >= 0.75:
                return True
        return False

    name = 'SIM A killer_dim descending'
    non_inv = [t for t in tokens if not t['is_invariant']]
    order = sorted(non_inv, key=lambda t: t['attractor_weights'][0], reverse=True)
    if not simulate([t['id'] for t in order]):
        fail(name, "did not converge by turn 18")

    name = 'SIM B locations first'
    locs = [t for t in non_inv if t['class']=='LOCATION']
    rest = [t for t in non_inv if t['class']!='LOCATION']
    order = locs + sorted(rest, key=lambda t: sum(t['attractor_weights']), reverse=True)
    if not simulate([t['id'] for t in order]):
        fail(name, "did not converge by turn 18")

    name = 'SIM C red herring then enabler'
    red = [t for t in non_inv if RED_TAGS.issubset(set(t['affinity_tags']))]
    # find enabler: suspect with k>=0.2 and m>=0.2
    enabler = None
    for t in non_inv:
        if t['class']=='SUSPECT' and t['attractor_weights'][0] >= 0.2 and t['attractor_weights'][1] >= 0.2:
            enabler = t
            break
    order = red + ([enabler] if enabler else []) + [t for t in non_inv if t not in red and t != enabler]
    if not simulate([t['id'] for t in order]):
        fail(name, "did not converge by turn 18")

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
