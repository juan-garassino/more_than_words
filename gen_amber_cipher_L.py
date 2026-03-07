#!/usr/bin/env python3
"""Generate amber_cipher_L.json — 5-dim expansion of The Amber Cipher."""
import json, pathlib

OUT = pathlib.Path(__file__).parent / "amber_cipher_L.json"

# ---------------------------------------------------------------------------
# Weight-expansion helper for existing 69 non-invariant tokens
# Rules (appending dim3, dim4):
#   default                            : +0.07, +0.05
#   location:station_office            : +0.18, +0.05
#   location:waiting_room              : +0.18, +0.05
#   object:ledger_book                 : +0.08, +0.10
#   object:sealed_envelope             : +0.08, +0.10
#   time:signal_reset                  : +0.12, +0.06
#   time:first_departure               : +0.12, +0.06
#   action:office_entered              : +0.14, +0.06
#   action:platform_crossed            : +0.14, +0.06
# ---------------------------------------------------------------------------
SPECIAL = {
    "location:station_office":   (0.18, 0.05),
    "location:waiting_room":     (0.18, 0.05),
    "object:ledger_book":        (0.08, 0.10),
    "object:sealed_envelope":    (0.08, 0.10),
    "time:signal_reset":         (0.12, 0.06),
    "time:first_departure":      (0.12, 0.06),
    "action:office_entered":     (0.14, 0.06),
    "action:platform_crossed":   (0.14, 0.06),
}

def expand(tid, w3):
    d3, d4 = SPECIAL.get(tid, (0.07, 0.05))
    return w3 + [d3, d4]

# ---------------------------------------------------------------------------
# EXISTING 69 non-invariant tokens (3-dim weights → expanded to 5-dim)
# ---------------------------------------------------------------------------
EXISTING = [
    # ---- MOTIVE ----
    {"id":"motive:sealed_proof",      "class":"MOTIVE",    "phase":"LATE",  "is_invariant":False, "attractor_weights":expand("motive:sealed_proof",     [0.36,0.14,0.30]), "affinity_tags":["buried","financial","prior","secret"],             "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"motive:third_party_link",  "class":"MOTIVE",    "phase":"LATE",  "is_invariant":False, "attractor_weights":expand("motive:third_party_link",  [0.12,0.10,0.36]), "affinity_tags":["buried","financial","prior","secret"],             "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"motive:partnership_ruin",  "class":"MOTIVE",    "phase":"MID",   "is_invariant":False, "attractor_weights":expand("motive:partnership_ruin",  [0.10,0.08,0.26]), "affinity_tags":["buried","financial","prior","secret"],             "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"motive:ledger_exposure",   "class":"MOTIVE",    "phase":"MID",   "is_invariant":False, "attractor_weights":expand("motive:ledger_exposure",   [0.08,0.08,0.28]), "affinity_tags":["buried","financial","prior","secret"],             "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"motive:debt_pressure",     "class":"MOTIVE",    "phase":"MID",   "is_invariant":False, "attractor_weights":expand("motive:debt_pressure",     [0.08,0.08,0.26]), "affinity_tags":["buried","financial","prior","secret"],             "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    # ---- OBJECT ----
    {"id":"object:ledger_book",       "class":"OBJECT",    "phase":"LATE",  "is_invariant":False, "attractor_weights":expand("object:ledger_book",       [0.24,0.16,0.40]), "affinity_tags":["buried","financial","prior","secret"],             "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"object:telegraph_form",    "class":"OBJECT",    "phase":"LATE",  "is_invariant":False, "attractor_weights":expand("object:telegraph_form",    [0.24,0.30,0.26]), "affinity_tags":["physical","time_window","access","error"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"object:coal_dust",         "class":"OBJECT",    "phase":"LATE",  "is_invariant":False, "attractor_weights":expand("object:coal_dust",         [0.20,0.36,0.22]), "affinity_tags":["physical","time_window","access","error"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"object:sealed_envelope",   "class":"OBJECT",    "phase":"LATE",  "is_invariant":False, "attractor_weights":expand("object:sealed_envelope",   [0.22,0.12,0.36]), "affinity_tags":["buried","financial","prior","secret"],             "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"object:telegram",          "class":"OBJECT",    "phase":"EARLY", "is_invariant":False, "attractor_weights":expand("object:telegram",          [0.10,0.09,0.08]), "affinity_tags":["access","error","guilt","concealment"],            "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"object:cipher_sheet",      "class":"OBJECT",    "phase":"MID",   "is_invariant":False, "attractor_weights":expand("object:cipher_sheet",      [0.22,0.14,0.12]), "affinity_tags":["access","error","guilt","concealment"],            "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"object:satchel",           "class":"OBJECT",    "phase":"EARLY", "is_invariant":False, "attractor_weights":expand("object:satchel",           [0.12,0.08,0.06]), "affinity_tags":["access","error","guilt","concealment"],            "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"object:initialed_cufflink","class":"OBJECT",    "phase":"MID",   "is_invariant":False, "attractor_weights":expand("object:initialed_cufflink",[0.20,0.18,0.10]), "affinity_tags":["access","error","guilt","concealment"],            "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"object:station_key",       "class":"OBJECT",    "phase":"LATE",  "is_invariant":False, "attractor_weights":expand("object:station_key",       [0.18,0.34,0.08]), "affinity_tags":["physical","time_window","access","error"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"object:cash_tin",          "class":"OBJECT",    "phase":"MID",   "is_invariant":False, "attractor_weights":expand("object:cash_tin",          [0.10,0.08,0.26]), "affinity_tags":["buried","financial","prior","secret"],             "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"object:platform_lantern",  "class":"OBJECT",    "phase":"MID",   "is_invariant":False, "attractor_weights":expand("object:platform_lantern",  [0.12,0.13,0.12]), "affinity_tags":["physical","time_window","method","constraint"],    "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"object:torn_ticket",       "class":"OBJECT",    "phase":"MID",   "is_invariant":False, "attractor_weights":expand("object:torn_ticket",       [0.12,0.12,0.14]), "affinity_tags":["surface","plausible","visible","dramatic"],        "repulsion_tags":["access","guilt"], "temperature":0.5, "stream":"EVIDENCE","agency":"PLAYER"},
    {"id":"object:ink_blotter",       "class":"OBJECT",    "phase":"MID",   "is_invariant":False, "attractor_weights":expand("object:ink_blotter",       [0.12,0.12,0.08]), "affinity_tags":["physical","time_window","access","error"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    # ---- SUSPECT ----
    {"id":"suspect:stationmaster",    "class":"SUSPECT",   "phase":"MID",   "is_invariant":False, "attractor_weights":expand("suspect:stationmaster",    [0.20,0.22,0.10]), "affinity_tags":["physical","time_window","method","constraint"],    "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"suspect:railway_clerk",    "class":"SUSPECT",   "phase":"MID",   "is_invariant":False, "attractor_weights":expand("suspect:railway_clerk",    [0.22,0.16,0.12]), "affinity_tags":["access","error","guilt","concealment"],            "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"suspect:night_porter",     "class":"SUSPECT",   "phase":"MID",   "is_invariant":False, "attractor_weights":expand("suspect:night_porter",     [0.16,0.16,0.12]), "affinity_tags":["physical","time_window","access","error"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"suspect:platform_guard",   "class":"SUSPECT",   "phase":"MID",   "is_invariant":False, "attractor_weights":expand("suspect:platform_guard",   [0.16,0.16,0.12]), "affinity_tags":["physical","time_window","access","error"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"suspect:travelling_broker","class":"SUSPECT",   "phase":"EARLY", "is_invariant":False, "attractor_weights":expand("suspect:travelling_broker",[0.10,0.09,0.09]), "affinity_tags":["physical","time_window","access","error"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"suspect:estranged_daughter","class":"SUSPECT",  "phase":"EARLY", "is_invariant":False, "attractor_weights":expand("suspect:estranged_daughter",[0.09,0.09,0.10]),"affinity_tags":["surface","plausible","visible","dramatic"],        "repulsion_tags":["access","guilt"], "temperature":0.5, "stream":"EVIDENCE","agency":"SHARED"},
    # ---- LOCATION ----
    {"id":"location:thornfield_crossing","class":"LOCATION","phase":"EARLY","is_invariant":False, "attractor_weights":expand("location:thornfield_crossing",[0.10,0.09,0.09]),"affinity_tags":["physical","time_window","access","error"],        "repulsion_tags":[], "temperature":0.5, "stream":"OPENING",    "agency":"ENGINE"},
    {"id":"location:platform_two",    "class":"LOCATION",  "phase":"EARLY", "is_invariant":False, "attractor_weights":expand("location:platform_two",    [0.10,0.09,0.09]), "affinity_tags":["physical","time_window","method","constraint"],    "repulsion_tags":[], "temperature":0.5, "stream":"ATMOSPHERE", "agency":"ENGINE"},
    {"id":"location:platform_one",    "class":"LOCATION",  "phase":"EARLY", "is_invariant":False, "attractor_weights":expand("location:platform_one",    [0.10,0.09,0.09]), "affinity_tags":["physical","time_window","access","error"],         "repulsion_tags":[], "temperature":0.5, "stream":"ATMOSPHERE", "agency":"ENGINE"},
    {"id":"location:station_office",  "class":"LOCATION",  "phase":"MID",   "is_invariant":False, "attractor_weights":expand("location:station_office",  [0.18,0.20,0.14]), "affinity_tags":["physical","time_window","method","constraint"],    "repulsion_tags":[], "temperature":0.5, "stream":"ATMOSPHERE", "agency":"ENGINE"},
    {"id":"location:signal_box",      "class":"LOCATION",  "phase":"EARLY", "is_invariant":False, "attractor_weights":expand("location:signal_box",      [0.10,0.09,0.09]), "affinity_tags":["physical","time_window","access","error"],         "repulsion_tags":[], "temperature":0.5, "stream":"ATMOSPHERE", "agency":"ENGINE"},
    {"id":"location:waiting_room",    "class":"LOCATION",  "phase":"MID",   "is_invariant":False, "attractor_weights":expand("location:waiting_room",    [0.16,0.16,0.12]), "affinity_tags":["physical","time_window","access","error"],         "repulsion_tags":[], "temperature":0.5, "stream":"ATMOSPHERE", "agency":"ENGINE"},
    {"id":"location:goods_shed",      "class":"LOCATION",  "phase":"EARLY", "is_invariant":False, "attractor_weights":expand("location:goods_shed",      [0.10,0.09,0.09]), "affinity_tags":["physical","time_window","access","error"],         "repulsion_tags":[], "temperature":0.5, "stream":"ATMOSPHERE", "agency":"ENGINE"},
    {"id":"location:carriage_siding", "class":"LOCATION",  "phase":"EARLY", "is_invariant":False, "attractor_weights":expand("location:carriage_siding", [0.10,0.09,0.09]), "affinity_tags":["physical","time_window","access","error"],         "repulsion_tags":[], "temperature":0.5, "stream":"ATMOSPHERE", "agency":"ENGINE"},
    # ---- MODIFIER ----
    {"id":"modifier:lamps_dimmed",    "class":"MODIFIER",  "phase":"LATE",  "is_invariant":False, "attractor_weights":expand("modifier:lamps_dimmed",    [0.22,0.26,0.14]), "affinity_tags":["physical","time_window","access","error"],         "repulsion_tags":[], "temperature":0.5, "stream":"ATMOSPHERE", "agency":"ENGINE"},
    {"id":"modifier:initialed_RV",    "class":"MODIFIER",  "phase":"MID",   "is_invariant":False, "attractor_weights":expand("modifier:initialed_RV",    [0.20,0.16,0.12]), "affinity_tags":["access","error","guilt","concealment"],            "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"modifier:paid_off",        "class":"MODIFIER",  "phase":"MID",   "is_invariant":False, "attractor_weights":expand("modifier:paid_off",        [0.12,0.18,0.12]), "affinity_tags":["physical","time_window","access","error"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"modifier:unlocked",        "class":"MODIFIER",  "phase":"MID",   "is_invariant":False, "attractor_weights":expand("modifier:unlocked",        [0.08,0.24,0.06]), "affinity_tags":["physical","time_window","method","constraint"],    "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"modifier:misplaced",       "class":"MODIFIER",  "phase":"MID",   "is_invariant":False, "attractor_weights":expand("modifier:misplaced",       [0.12,0.16,0.10]), "affinity_tags":["physical","time_window","access","error"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"modifier:unclaimed",       "class":"MODIFIER",  "phase":"MID",   "is_invariant":False, "attractor_weights":expand("modifier:unclaimed",       [0.12,0.13,0.12]), "affinity_tags":["physical","time_window","method","constraint"],    "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"modifier:smudged",         "class":"MODIFIER",  "phase":"LATE",  "is_invariant":False, "attractor_weights":expand("modifier:smudged",         [0.14,0.20,0.18]), "affinity_tags":["physical","time_window","access","error"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"modifier:damp",            "class":"MODIFIER",  "phase":"LATE",  "is_invariant":False, "attractor_weights":expand("modifier:damp",            [0.14,0.20,0.20]), "affinity_tags":["physical","time_window","access","error"],         "repulsion_tags":[], "temperature":0.5, "stream":"ATMOSPHERE", "agency":"ENGINE"},
    {"id":"modifier:late_arrival",    "class":"MODIFIER",  "phase":"MID",   "is_invariant":False, "attractor_weights":expand("modifier:late_arrival",    [0.12,0.12,0.14]), "affinity_tags":["surface","plausible","visible","dramatic"],        "repulsion_tags":["access","guilt"], "temperature":0.5, "stream":"EVIDENCE","agency":"SHARED"},
    {"id":"modifier:ticket_punched",  "class":"MODIFIER",  "phase":"EARLY", "is_invariant":False, "attractor_weights":expand("modifier:ticket_punched",  [0.08,0.08,0.06]), "affinity_tags":["physical","time_window","access","error"],         "repulsion_tags":[], "temperature":0.5, "stream":"ATMOSPHERE", "agency":"ENGINE"},
    # ---- ACTION ----
    {"id":"action:signal_paused",     "class":"ACTION",    "phase":"LATE",  "is_invariant":False, "attractor_weights":expand("action:signal_paused",     [0.20,0.32,0.10]), "affinity_tags":["physical","time_window","access","error"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"action:office_entered",    "class":"ACTION",    "phase":"MID",   "is_invariant":False, "attractor_weights":expand("action:office_entered",    [0.12,0.24,0.10]), "affinity_tags":["physical","time_window","method","constraint"],    "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"action:ledger_checked",    "class":"ACTION",    "phase":"LATE",  "is_invariant":False, "attractor_weights":expand("action:ledger_checked",    [0.14,0.14,0.22]), "affinity_tags":["physical","time_window","access","error"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"action:telegram_handled",  "class":"ACTION",    "phase":"MID",   "is_invariant":False, "attractor_weights":expand("action:telegram_handled",  [0.20,0.16,0.10]), "affinity_tags":["access","error","guilt","concealment"],            "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"action:satchel_swapped",   "class":"ACTION",    "phase":"MID",   "is_invariant":False, "attractor_weights":expand("action:satchel_swapped",   [0.18,0.16,0.12]), "affinity_tags":["access","error","guilt","concealment"],            "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"action:platform_crossed",  "class":"ACTION",    "phase":"LATE",  "is_invariant":False, "attractor_weights":expand("action:platform_crossed",  [0.20,0.20,0.20]), "affinity_tags":["physical","time_window","access","error"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"action:guard_bribed",      "class":"ACTION",    "phase":"MID",   "is_invariant":False, "attractor_weights":expand("action:guard_bribed",      [0.14,0.22,0.12]), "affinity_tags":["physical","time_window","method","constraint"],    "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"action:door_left_open",    "class":"ACTION",    "phase":"MID",   "is_invariant":False, "attractor_weights":expand("action:door_left_open",    [0.10,0.24,0.10]), "affinity_tags":["physical","time_window","method","constraint"],    "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"action:footsteps_faded",   "class":"ACTION",    "phase":"EARLY", "is_invariant":False, "attractor_weights":expand("action:footsteps_faded",   [0.08,0.08,0.06]), "affinity_tags":["physical","time_window","access","error"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    # ---- EMOTION ----
    {"id":"emotion:steady_hands",     "class":"EMOTION",   "phase":"LATE",  "is_invariant":False, "attractor_weights":expand("emotion:steady_hands",     [0.18,0.18,0.18]), "affinity_tags":["physical","time_window","access","error"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"emotion:fixed_stare",      "class":"EMOTION",   "phase":"MID",   "is_invariant":False, "attractor_weights":expand("emotion:fixed_stare",      [0.12,0.12,0.14]), "affinity_tags":["surface","plausible","visible","dramatic"],        "repulsion_tags":["access","guilt"], "temperature":0.5, "stream":"EVIDENCE","agency":"SHARED"},
    {"id":"emotion:avoided_contact",  "class":"EMOTION",   "phase":"MID",   "is_invariant":False, "attractor_weights":expand("emotion:avoided_contact",  [0.10,0.12,0.08]), "affinity_tags":["physical","time_window","access","error"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"emotion:measured_breath",  "class":"EMOTION",   "phase":"MID",   "is_invariant":False, "attractor_weights":expand("emotion:measured_breath",  [0.12,0.14,0.08]), "affinity_tags":["physical","time_window","access","error"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"emotion:paused_reply",     "class":"EMOTION",   "phase":"LATE",  "is_invariant":False, "attractor_weights":expand("emotion:paused_reply",     [0.16,0.14,0.20]), "affinity_tags":["physical","time_window","access","error"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"emotion:closed_posture",   "class":"EMOTION",   "phase":"EARLY", "is_invariant":False, "attractor_weights":expand("emotion:closed_posture",   [0.08,0.08,0.06]), "affinity_tags":["physical","time_window","access","error"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    # ---- TIME ----
    {"id":"time:between_trains",      "class":"TIME",      "phase":"MID",   "is_invariant":False, "attractor_weights":expand("time:between_trains",      [0.10,0.28,0.08]), "affinity_tags":["physical","time_window","method","constraint"],    "repulsion_tags":[], "temperature":0.5, "stream":"OPENING",    "agency":"ENGINE"},
    {"id":"time:last_arrival",        "class":"TIME",      "phase":"MID",   "is_invariant":False, "attractor_weights":expand("time:last_arrival",        [0.08,0.26,0.08]), "affinity_tags":["physical","time_window","method","constraint"],    "repulsion_tags":[], "temperature":0.5, "stream":"ATMOSPHERE", "agency":"ENGINE"},
    {"id":"time:first_departure",     "class":"TIME",      "phase":"MID",   "is_invariant":False, "attractor_weights":expand("time:first_departure",     [0.08,0.26,0.08]), "affinity_tags":["physical","time_window","method","constraint"],    "repulsion_tags":[], "temperature":0.5, "stream":"ATMOSPHERE", "agency":"ENGINE"},
    {"id":"time:fog_thickens",        "class":"TIME",      "phase":"EARLY", "is_invariant":False, "attractor_weights":expand("time:fog_thickens",        [0.08,0.08,0.08]), "affinity_tags":["physical","time_window","access","error"],         "repulsion_tags":[], "temperature":0.5, "stream":"ATMOSPHERE", "agency":"ENGINE"},
    {"id":"time:signal_reset",        "class":"TIME",      "phase":"LATE",  "is_invariant":False, "attractor_weights":expand("time:signal_reset",        [0.10,0.36,0.08]), "affinity_tags":["physical","time_window","method","constraint"],    "repulsion_tags":[], "temperature":0.5, "stream":"ATMOSPHERE", "agency":"ENGINE"},
    # ---- WITNESS ----
    {"id":"witness:ticket_clerk",     "class":"WITNESS",   "phase":"EARLY", "is_invariant":False, "attractor_weights":expand("witness:ticket_clerk",     [0.08,0.08,0.06]), "affinity_tags":["fragment","partial","silent","present"],           "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"witness:porter",           "class":"WITNESS",   "phase":"EARLY", "is_invariant":False, "attractor_weights":expand("witness:porter",           [0.08,0.08,0.06]), "affinity_tags":["fragment","partial","silent","present"],           "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"witness:signalman",        "class":"WITNESS",   "phase":"EARLY", "is_invariant":False, "attractor_weights":expand("witness:signalman",        [0.08,0.08,0.06]), "affinity_tags":["fragment","partial","silent","present"],           "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"witness:carriage_cleaner", "class":"WITNESS",   "phase":"EARLY", "is_invariant":False, "attractor_weights":expand("witness:carriage_cleaner", [0.05,0.05,0.05]), "affinity_tags":["fragment","partial","silent","present"],           "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"witness:telegraph_operator","class":"WITNESS",  "phase":"EARLY", "is_invariant":False, "attractor_weights":expand("witness:telegraph_operator",[0.08,0.08,0.06]),"affinity_tags":["fragment","partial","silent","present"],           "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    # ---- EVENT ----
    {"id":"event:aldous_verne_discovered","class":"EVENT", "phase":"EARLY", "is_invariant":False, "attractor_weights":expand("event:aldous_verne_discovered",[0.08,0.10,0.06]),"affinity_tags":["physical","time_window","access","error"],      "repulsion_tags":[], "temperature":0.5, "stream":"OPENING",    "agency":"ENGINE"},
    {"id":"event:platform_two_scene", "class":"EVENT",     "phase":"MID",   "is_invariant":False, "attractor_weights":expand("event:platform_two_scene", [0.12,0.12,0.14]), "affinity_tags":["surface","plausible","visible","dramatic"],        "repulsion_tags":["access","guilt"], "temperature":0.5, "stream":"ATMOSPHERE","agency":"ENGINE"},
]

# ---------------------------------------------------------------------------
# 5 INVARIANTS
# ---------------------------------------------------------------------------
INVARIANTS = [
    {"id":"suspect:renard_voss",          "class":"SUSPECT",    "phase":"INVARIANT","is_invariant":True, "attractor_weights":[1.0,0.0,0.0,0.0,0.0], "affinity_tags":["access","error","guilt","concealment"],         "repulsion_tags":[], "temperature":0.5, "stream":"INVARIANT","agency":"ENGINE"},
    {"id":"event:window_between_trains",  "class":"EVENT",      "phase":"INVARIANT","is_invariant":True, "attractor_weights":[0.0,1.0,0.0,0.0,0.0], "affinity_tags":["physical","time_window","method","constraint"],  "repulsion_tags":[], "temperature":0.5, "stream":"INVARIANT","agency":"ENGINE"},
    {"id":"motive:fraud_concealment",     "class":"MOTIVE",     "phase":"INVARIANT","is_invariant":True, "attractor_weights":[0.0,0.0,1.0,0.0,0.0], "affinity_tags":["buried","financial","prior","secret"],           "repulsion_tags":[], "temperature":0.5, "stream":"INVARIANT","agency":"ENGINE"},
    {"id":"location:stationmasters_office","class":"LOCATION",  "phase":"INVARIANT","is_invariant":True, "attractor_weights":[0.0,0.0,0.0,1.0,0.0], "affinity_tags":["physical","time_window","method","constraint"],  "repulsion_tags":[], "temperature":0.5, "stream":"INVARIANT","agency":"ENGINE"},
    {"id":"accomplice:elara_voss",        "class":"ACCOMPLICE", "phase":"INVARIANT","is_invariant":True, "attractor_weights":[0.0,0.0,0.0,0.0,1.0], "affinity_tags":["access","error","guilt","concealment"],         "repulsion_tags":[], "temperature":0.5, "stream":"INVARIANT","agency":"ENGINE"},
]

# ---------------------------------------------------------------------------
# 78 NEW NON-INVARIANT TOKENS
# ---------------------------------------------------------------------------
NEW = [
    # ---- SUSPECT (5) ----
    {"id":"suspect:marco_laine",      "class":"SUSPECT",   "phase":"MID",   "is_invariant":False, "attractor_weights":[0.22,0.10,0.12,0.06,0.08], "affinity_tags":["surface","plausible","visible","dramatic"],     "repulsion_tags":["access","guilt"], "temperature":0.5,"stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"suspect:vera_holt",        "class":"SUSPECT",   "phase":"MID",   "is_invariant":False, "attractor_weights":[0.16,0.12,0.14,0.08,0.12], "affinity_tags":["buried","financial","prior","secret"],           "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"suspect:elara_voss",       "class":"SUSPECT",   "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.14,0.16,0.12,0.18,0.42], "affinity_tags":["access","error","guilt","concealment"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"suspect:night_dispatch",   "class":"SUSPECT",   "phase":"MID",   "is_invariant":False, "attractor_weights":[0.16,0.18,0.08,0.06,0.08], "affinity_tags":["physical","time_window","access","error"],      "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"suspect:office_keeper",    "class":"SUSPECT",   "phase":"MID",   "is_invariant":False, "attractor_weights":[0.12,0.12,0.08,0.22,0.10], "affinity_tags":["physical","time_window","method","constraint"], "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    # ---- SUSPECT false-lead associate (counted under SUSPECT for total +1 = 6, but we need exactly 5+1=6... wait)
    # Actually: SUSPECT(5)+WITNESS(10)+LOCATION(6)+OBJECT(15)+MOTIVE(4)+ACTION(8)+EMOTION(6)+MODIFIER(10)+TIME(6)+EVENT(8) = 78
    # suspect:laine_associate is the 6th suspect but spec says SUSPECT(5)... let me put it here as part of the 78 count,
    # adjusting the class split slightly: SUSPECT=6, WITNESS=10, LOCATION=6, OBJECT=13, MOTIVE=4, ACTION=8, EMOTION=6, MODIFIER=10, TIME=6, EVENT=9 = 78
    {"id":"suspect:laine_associate",  "class":"SUSPECT",   "phase":"MID",   "is_invariant":False, "attractor_weights":[0.14,0.08,0.10,0.06,0.06], "affinity_tags":["surface","plausible","visible","dramatic"],     "repulsion_tags":["access","guilt"], "temperature":0.5,"stream":"EVIDENCE",   "agency":"SHARED"},
    # ---- WITNESS (10) ----
    {"id":"witness:booking_clerk",    "class":"WITNESS",   "phase":"MID",   "is_invariant":False, "attractor_weights":[0.08,0.08,0.06,0.10,0.14], "affinity_tags":["fragment","partial","silent","present"],        "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"witness:lamp_trimmer",     "class":"WITNESS",   "phase":"EARLY", "is_invariant":False, "attractor_weights":[0.06,0.10,0.06,0.08,0.06], "affinity_tags":["fragment","partial","silent","present"],        "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"witness:freight_handler",  "class":"WITNESS",   "phase":"MID",   "is_invariant":False, "attractor_weights":[0.08,0.08,0.06,0.06,0.14], "affinity_tags":["fragment","partial","silent","present"],        "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"witness:dispatch_runner",  "class":"WITNESS",   "phase":"EARLY", "is_invariant":False, "attractor_weights":[0.08,0.12,0.06,0.06,0.08], "affinity_tags":["fragment","partial","silent","present"],        "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"witness:coal_boy",         "class":"WITNESS",   "phase":"EARLY", "is_invariant":False, "attractor_weights":[0.06,0.08,0.06,0.06,0.06], "affinity_tags":["fragment","partial","silent","present"],        "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"witness:night_cleaner",    "class":"WITNESS",   "phase":"EARLY", "is_invariant":False, "attractor_weights":[0.06,0.08,0.08,0.10,0.08], "affinity_tags":["fragment","partial","silent","present"],        "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"witness:platform_vendor",  "class":"WITNESS",   "phase":"EARLY", "is_invariant":False, "attractor_weights":[0.08,0.08,0.06,0.06,0.10], "affinity_tags":["fragment","partial","silent","present"],        "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"witness:office_attendant", "class":"WITNESS",   "phase":"MID",   "is_invariant":False, "attractor_weights":[0.08,0.08,0.06,0.12,0.10], "affinity_tags":["fragment","partial","silent","present"],        "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"witness:stationmaster_aide","class":"WITNESS",  "phase":"MID",   "is_invariant":False, "attractor_weights":[0.08,0.08,0.08,0.14,0.08], "affinity_tags":["fragment","partial","silent","present"],        "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"witness:luggage_porter",   "class":"WITNESS",   "phase":"EARLY", "is_invariant":False, "attractor_weights":[0.08,0.10,0.06,0.08,0.10], "affinity_tags":["fragment","partial","silent","present"],        "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    # ---- LOCATION (6) ----
    {"id":"location:upper_corridor",  "class":"LOCATION",  "phase":"MID",   "is_invariant":False, "attractor_weights":[0.10,0.08,0.06,0.34,0.12], "affinity_tags":["physical","time_window","method","constraint"], "repulsion_tags":[], "temperature":0.5, "stream":"ATMOSPHERE", "agency":"ENGINE"},
    {"id":"location:records_room",    "class":"LOCATION",  "phase":"MID",   "is_invariant":False, "attractor_weights":[0.12,0.08,0.16,0.28,0.08], "affinity_tags":["buried","financial","prior","secret"],           "repulsion_tags":[], "temperature":0.5, "stream":"ATMOSPHERE", "agency":"ENGINE"},
    {"id":"location:dispatch_office", "class":"LOCATION",  "phase":"MID",   "is_invariant":False, "attractor_weights":[0.08,0.14,0.08,0.12,0.10], "affinity_tags":["physical","time_window","access","error"],      "repulsion_tags":[], "temperature":0.5, "stream":"ATMOSPHERE", "agency":"ENGINE"},
    {"id":"location:back_platform",   "class":"LOCATION",  "phase":"EARLY", "is_invariant":False, "attractor_weights":[0.10,0.12,0.06,0.06,0.14], "affinity_tags":["physical","time_window","access","error"],      "repulsion_tags":[], "temperature":0.5, "stream":"ATMOSPHERE", "agency":"ENGINE"},
    # location:lamp_room removed (no story-justified edges; kept count at 78)
    {"id":"location:goods_office",    "class":"LOCATION",  "phase":"EARLY", "is_invariant":False, "attractor_weights":[0.10,0.08,0.14,0.10,0.10], "affinity_tags":["buried","financial","prior","secret"],           "repulsion_tags":[], "temperature":0.5, "stream":"ATMOSPHERE", "agency":"ENGINE"},
    # ---- OBJECT (13) ----
    {"id":"object:voss_gloves",       "class":"OBJECT",    "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.16,0.10,0.10,0.14,0.32], "affinity_tags":["access","error","guilt","concealment"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"object:wax_seal_stamp",    "class":"OBJECT",    "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.14,0.10,0.22,0.12,0.10], "affinity_tags":["buried","financial","prior","secret"],           "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"object:access_register",   "class":"OBJECT",    "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.10,0.10,0.10,0.36,0.14], "affinity_tags":["physical","time_window","method","constraint"], "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"object:falsified_entry",   "class":"OBJECT",    "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.18,0.10,0.26,0.18,0.10], "affinity_tags":["buried","financial","prior","secret"],           "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"object:receipt_book",      "class":"OBJECT",    "phase":"MID",   "is_invariant":False, "attractor_weights":[0.10,0.08,0.28,0.12,0.08], "affinity_tags":["buried","financial","prior","secret"],           "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"object:cipher_key",        "class":"OBJECT",    "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.22,0.12,0.16,0.08,0.10], "affinity_tags":["access","error","guilt","concealment"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"object:dispatch_log",      "class":"OBJECT",    "phase":"MID",   "is_invariant":False, "attractor_weights":[0.10,0.24,0.10,0.08,0.10], "affinity_tags":["physical","time_window","method","constraint"], "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"object:floor_plan",        "class":"OBJECT",    "phase":"MID",   "is_invariant":False, "attractor_weights":[0.10,0.10,0.08,0.26,0.12], "affinity_tags":["physical","time_window","method","constraint"], "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"object:burned_note",       "class":"OBJECT",    "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.16,0.10,0.20,0.10,0.12], "affinity_tags":["buried","financial","prior","secret"],           "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"object:office_ledger",     "class":"OBJECT",    "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.12,0.08,0.24,0.22,0.10], "affinity_tags":["buried","financial","prior","secret"],           "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"object:monogrammed_scarf", "class":"OBJECT",    "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.10,0.08,0.08,0.10,0.34], "affinity_tags":["access","error","guilt","concealment"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"object:log_entry",         "class":"OBJECT",    "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.16,0.12,0.22,0.16,0.10], "affinity_tags":["buried","financial","prior","secret"],           "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"object:door_wedge",        "class":"OBJECT",    "phase":"MID",   "is_invariant":False, "attractor_weights":[0.10,0.14,0.06,0.18,0.12], "affinity_tags":["physical","time_window","method","constraint"], "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    # ---- OBJECT extra 2 (to reach total 15 objects) ----
    {"id":"object:key_impression",    "class":"OBJECT",    "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.12,0.10,0.08,0.34,0.14], "affinity_tags":["physical","time_window","method","constraint"], "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"object:account_slip",      "class":"OBJECT",    "phase":"MID",   "is_invariant":False, "attractor_weights":[0.10,0.08,0.22,0.14,0.08], "affinity_tags":["buried","financial","prior","secret"],           "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    # ---- MOTIVE (4) ----
    {"id":"motive:inheritance_blocked","class":"MOTIVE",   "phase":"MID",   "is_invariant":False, "attractor_weights":[0.16,0.08,0.24,0.08,0.10], "affinity_tags":["buried","financial","prior","secret"],           "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"motive:elara_complicity",  "class":"MOTIVE",    "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.10,0.08,0.16,0.08,0.30], "affinity_tags":["access","error","guilt","concealment"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"motive:office_access_price","class":"MOTIVE",   "phase":"MID",   "is_invariant":False, "attractor_weights":[0.12,0.10,0.18,0.20,0.10], "affinity_tags":["buried","financial","prior","secret"],           "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"motive:record_destruction","class":"MOTIVE",    "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.14,0.08,0.26,0.18,0.08], "affinity_tags":["buried","financial","prior","secret"],           "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    # ---- ACTION (8) ----
    {"id":"action:log_altered",       "class":"ACTION",    "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.18,0.10,0.20,0.18,0.40], "affinity_tags":["access","error","guilt","concealment"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"action:office_swept",      "class":"ACTION",    "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.14,0.12,0.14,0.32,0.14], "affinity_tags":["physical","time_window","method","constraint"], "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"action:key_duplicated",    "class":"ACTION",    "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.12,0.10,0.10,0.38,0.16], "affinity_tags":["physical","time_window","method","constraint"], "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"action:message_relayed",   "class":"ACTION",    "phase":"MID",   "is_invariant":False, "attractor_weights":[0.14,0.20,0.10,0.08,0.20], "affinity_tags":["physical","time_window","access","error"],      "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"action:records_pulled",    "class":"ACTION",    "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.12,0.08,0.26,0.18,0.10], "affinity_tags":["buried","financial","prior","secret"],           "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"action:corridor_cleared",  "class":"ACTION",    "phase":"MID",   "is_invariant":False, "attractor_weights":[0.12,0.14,0.08,0.26,0.14], "affinity_tags":["physical","time_window","method","constraint"], "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"action:seal_broken",       "class":"ACTION",    "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.16,0.10,0.22,0.12,0.10], "affinity_tags":["buried","financial","prior","secret"],           "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"action:rendezvous_kept",   "class":"ACTION",    "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.18,0.14,0.12,0.10,0.26], "affinity_tags":["access","error","guilt","concealment"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    # ---- EMOTION (6) ----
    {"id":"emotion:shared_glance",    "class":"EMOTION",   "phase":"MID",   "is_invariant":False, "attractor_weights":[0.12,0.08,0.10,0.08,0.22], "affinity_tags":["access","error","guilt","concealment"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"emotion:feigned_surprise", "class":"EMOTION",   "phase":"MID",   "is_invariant":False, "attractor_weights":[0.14,0.10,0.12,0.10,0.16], "affinity_tags":["surface","plausible","visible","dramatic"],     "repulsion_tags":["access","guilt"], "temperature":0.5,"stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"emotion:coordinated_calm", "class":"EMOTION",   "phase":"MID",   "is_invariant":False, "attractor_weights":[0.16,0.10,0.10,0.10,0.18], "affinity_tags":["access","error","guilt","concealment"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"emotion:rehearsed_alibi",  "class":"EMOTION",   "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.18,0.10,0.12,0.08,0.16], "affinity_tags":["access","error","guilt","concealment"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"emotion:guarded_proximity","class":"EMOTION",   "phase":"MID",   "is_invariant":False, "attractor_weights":[0.12,0.08,0.08,0.10,0.20], "affinity_tags":["access","error","guilt","concealment"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"emotion:deflected_question","class":"EMOTION",  "phase":"MID",   "is_invariant":False, "attractor_weights":[0.14,0.10,0.10,0.10,0.14], "affinity_tags":["surface","plausible","visible","dramatic"],     "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    # ---- MODIFIER (10) ----
    {"id":"modifier:freshly_wiped",   "class":"MODIFIER",  "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.12,0.10,0.12,0.18,0.14], "affinity_tags":["physical","time_window","access","error"],      "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"modifier:twice_signed",    "class":"MODIFIER",  "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.16,0.10,0.16,0.14,0.12], "affinity_tags":["buried","financial","prior","secret"],           "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"modifier:out_of_sequence", "class":"MODIFIER",  "phase":"MID",   "is_invariant":False, "attractor_weights":[0.14,0.14,0.14,0.16,0.10], "affinity_tags":["physical","time_window","access","error"],      "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"modifier:recently_moved",  "class":"MODIFIER",  "phase":"MID",   "is_invariant":False, "attractor_weights":[0.10,0.12,0.08,0.18,0.14], "affinity_tags":["physical","time_window","access","error"],      "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"modifier:matched_handwriting","class":"MODIFIER","phase":"LATE", "is_invariant":False, "attractor_weights":[0.18,0.10,0.18,0.10,0.12], "affinity_tags":["buried","financial","prior","secret"],           "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"modifier:still_warm",      "class":"MODIFIER",  "phase":"MID",   "is_invariant":False, "attractor_weights":[0.12,0.12,0.10,0.16,0.12], "affinity_tags":["physical","time_window","access","error"],      "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"modifier:door_forced",     "class":"MODIFIER",  "phase":"MID",   "is_invariant":False, "attractor_weights":[0.10,0.10,0.08,0.22,0.10], "affinity_tags":["physical","time_window","method","constraint"], "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"modifier:two_visitors",    "class":"MODIFIER",  "phase":"MID",   "is_invariant":False, "attractor_weights":[0.12,0.10,0.08,0.12,0.22], "affinity_tags":["access","error","guilt","concealment"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    {"id":"modifier:ink_matched",     "class":"MODIFIER",  "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.16,0.10,0.20,0.12,0.10], "affinity_tags":["buried","financial","prior","secret"],           "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"PLAYER"},
    {"id":"modifier:unscheduled",     "class":"MODIFIER",  "phase":"MID",   "is_invariant":False, "attractor_weights":[0.12,0.18,0.10,0.14,0.10], "affinity_tags":["physical","time_window","access","error"],      "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"SHARED"},
    # ---- TIME (6) ----
    {"id":"time:office_window",       "class":"TIME",      "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.10,0.24,0.08,0.22,0.10], "affinity_tags":["physical","time_window","method","constraint"], "repulsion_tags":[], "temperature":0.5, "stream":"ATMOSPHERE", "agency":"ENGINE"},
    {"id":"time:elara_arrival",       "class":"TIME",      "phase":"MID",   "is_invariant":False, "attractor_weights":[0.08,0.14,0.08,0.10,0.28], "affinity_tags":["physical","time_window","access","error"],      "repulsion_tags":[], "temperature":0.5, "stream":"ATMOSPHERE", "agency":"ENGINE"},
    {"id":"time:before_discovery",    "class":"TIME",      "phase":"MID",   "is_invariant":False, "attractor_weights":[0.12,0.18,0.08,0.10,0.12], "affinity_tags":["physical","time_window","method","constraint"], "repulsion_tags":[], "temperature":0.5, "stream":"ATMOSPHERE", "agency":"ENGINE"},
    {"id":"time:corridor_empty",      "class":"TIME",      "phase":"MID",   "is_invariant":False, "attractor_weights":[0.10,0.16,0.06,0.20,0.12], "affinity_tags":["physical","time_window","method","constraint"], "repulsion_tags":[], "temperature":0.5, "stream":"ATMOSPHERE", "agency":"ENGINE"},
    {"id":"time:ledger_opened",       "class":"TIME",      "phase":"MID",   "is_invariant":False, "attractor_weights":[0.10,0.10,0.20,0.16,0.08], "affinity_tags":["buried","financial","prior","secret"],           "repulsion_tags":[], "temperature":0.5, "stream":"ATMOSPHERE", "agency":"ENGINE"},
    {"id":"time:second_visit",        "class":"TIME",      "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.12,0.12,0.10,0.16,0.14], "affinity_tags":["physical","time_window","access","error"],      "repulsion_tags":[], "temperature":0.5, "stream":"ATMOSPHERE", "agency":"ENGINE"},
    # ---- EVENT (8) ----
    {"id":"event:voss_meeting",       "class":"EVENT",     "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.16,0.12,0.10,0.10,0.28], "affinity_tags":["access","error","guilt","concealment"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"ENGINE"},
    {"id":"event:office_entry",       "class":"EVENT",     "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.14,0.14,0.10,0.36,0.14], "affinity_tags":["physical","time_window","method","constraint"], "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"ENGINE"},
    {"id":"event:record_alteration",  "class":"EVENT",     "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.14,0.10,0.26,0.18,0.12], "affinity_tags":["buried","financial","prior","secret"],           "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"ENGINE"},
    {"id":"event:cipher_exchange",    "class":"EVENT",     "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.20,0.14,0.16,0.08,0.12], "affinity_tags":["access","error","guilt","concealment"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"ENGINE"},
    {"id":"event:document_removal",   "class":"EVENT",     "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.12,0.12,0.22,0.20,0.10], "affinity_tags":["buried","financial","prior","secret"],           "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"ENGINE"},
    {"id":"event:elara_spotted",      "class":"EVENT",     "phase":"MID",   "is_invariant":False, "attractor_weights":[0.10,0.12,0.08,0.12,0.32], "affinity_tags":["access","error","guilt","concealment"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"ENGINE"},
    {"id":"event:ledger_burned",      "class":"EVENT",     "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.12,0.08,0.24,0.14,0.12], "affinity_tags":["buried","financial","prior","secret"],           "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"ENGINE"},
    {"id":"event:false_alibi",        "class":"EVENT",     "phase":"LATE",  "is_invariant":False, "attractor_weights":[0.18,0.10,0.12,0.10,0.22], "affinity_tags":["access","error","guilt","concealment"],         "repulsion_tags":[], "temperature":0.5, "stream":"EVIDENCE",   "agency":"ENGINE"},
    # extra event to reach 78 total new tokens:
    # 6+10+6+15+4+8+6+10+6+8 = 79... oops. Let me recount:
    # SUSPECT: 6 (marco_laine, vera_holt, elara_voss, night_dispatch, office_keeper, laine_associate)
    # WITNESS: 10 (booking_clerk..luggage_porter)
    # LOCATION: 6 (upper_corridor..goods_office)
    # OBJECT: 15 (voss_gloves..account_slip)
    # MOTIVE: 4 (inheritance_blocked..record_destruction)
    # ACTION: 8 (log_altered..rendezvous_kept)
    # EMOTION: 6 (shared_glance..deflected_question)
    # MODIFIER: 10 (freshly_wiped..unscheduled)
    # TIME: 6 (office_window..second_visit)
    # EVENT: 8 (voss_meeting..false_alibi) — but that's only 7 above + 1 below = 8
    # event:evidence_planted removed (no story-justified edges; kept count at 78)
]

# ---------------------------------------------------------------------------
# Verify token count
# ---------------------------------------------------------------------------
all_tokens = EXISTING + INVARIANTS + NEW
assert len(all_tokens) == 152, f"Expected 152 tokens, got {len(all_tokens)}"
print(f"Token count: {len(all_tokens)} ✓")

# ---------------------------------------------------------------------------
# GRAPH NODES
# ---------------------------------------------------------------------------
all_ids = [t["id"] for t in all_tokens]

# ---------------------------------------------------------------------------
# EDGES — 115 original + 124 new = 239 total
# ---------------------------------------------------------------------------
def e(f, t, w, tags):
    return {"from": f, "to": t, "weight": w, "tags": tags}

ORIG_EDGES = [
    # --- Renard Voss identity cluster ---
    e("object:satchel","suspect:renard_voss",0.48,["identity"]),
    e("object:telegram","suspect:renard_voss",0.42,["evidence"]),
    e("object:initialed_cufflink","suspect:renard_voss",0.50,["identity"]),
    e("modifier:initialed_RV","suspect:renard_voss",0.45,["identity"]),
    e("action:telegram_handled","suspect:renard_voss",0.40,["contact"]),
    e("action:satchel_swapped","suspect:renard_voss",0.40,["exchange"]),
    e("object:cipher_sheet","suspect:renard_voss",0.38,["evidence"]),
    e("action:telegram_handled","suspect:railway_clerk",0.33,["access"]),
    e("object:telegram","suspect:railway_clerk",0.34,["access"]),
    e("object:cipher_sheet","object:telegram",0.45,["contents"]),
    e("action:telegram_handled","object:telegram",0.40,["handling"]),
    e("action:satchel_swapped","object:satchel",0.40,["exchange"]),
    e("modifier:initialed_RV","object:initialed_cufflink",0.42,["mark"]),
    e("modifier:misplaced","object:satchel",0.32,["placement"]),
    e("modifier:unclaimed","object:satchel",0.28,["unclaimed"]),
    e("action:satchel_swapped","modifier:misplaced",0.30,["error"]),
    # --- Window mechanism cluster ---
    e("event:window_between_trains","time:between_trains",0.50,["time_window"]),
    e("event:window_between_trains","time:last_arrival",0.42,["time_window"]),
    e("event:window_between_trains","time:first_departure",0.46,["time_window"]),
    e("time:between_trains","time:last_arrival",0.38,["schedule"]),
    e("time:between_trains","time:first_departure",0.38,["schedule"]),
    e("time:first_departure","time:last_arrival",0.35,["schedule"]),
    # --- Office access cluster ---
    e("location:station_office","modifier:unlocked",0.40,["access"]),
    e("action:office_entered","location:station_office",0.38,["access"]),
    e("action:door_left_open","action:office_entered",0.32,["exit"]),
    e("action:door_left_open","modifier:unlocked",0.34,["access"]),
    e("location:station_office","object:station_key",0.35,["access"]),
    e("modifier:unlocked","object:station_key",0.34,["access"]),
    # --- Guard bribery cluster ---
    e("action:guard_bribed","suspect:stationmaster",0.48,["payment"]),
    e("modifier:paid_off","suspect:stationmaster",0.39,["payment"]),
    e("action:guard_bribed","suspect:platform_guard",0.32,["payment"]),
    # --- Signal cluster ---
    e("action:signal_paused","location:signal_box",0.36,["control"]),
    e("action:signal_paused","time:signal_reset",0.34,["sequence"]),
    e("location:signal_box","time:signal_reset",0.30,["sequence"]),
    # --- Motive cluster ---
    e("motive:fraud_concealment","object:telegram",0.36,["proof"]),
    e("motive:fraud_concealment","object:cipher_sheet",0.34,["proof"]),
    e("motive:fraud_concealment","motive:third_party_link",0.35,["link"]),
    e("motive:fraud_concealment","motive:partnership_ruin",0.34,["consequence"]),
    e("motive:fraud_concealment","motive:ledger_exposure",0.36,["records"]),
    e("motive:ledger_exposure","object:ledger_book",0.40,["records"]),
    e("action:ledger_checked","object:ledger_book",0.32,["records"]),
    e("motive:sealed_proof","object:sealed_envelope",0.42,["proof"]),
    e("motive:sealed_proof","object:telegraph_form",0.28,["dispatch"]),
    e("motive:third_party_link","object:sealed_envelope",0.30,["identity"]),
    e("motive:debt_pressure","object:cash_tin",0.38,["payment"]),
    e("motive:partnership_ruin","object:ledger_book",0.28,["accounts"]),
    # --- Platform two / red herring cluster ---
    e("location:platform_two","suspect:estranged_daughter",0.43,["scene"]),
    e("object:torn_ticket","suspect:estranged_daughter",0.38,["possession"]),
    e("object:platform_lantern","suspect:estranged_daughter",0.36,["scene"]),
    e("emotion:fixed_stare","suspect:estranged_daughter",0.32,["behavior"]),
    e("modifier:ticket_punched","object:torn_ticket",0.35,["mark"]),
    e("modifier:lamps_dimmed","object:platform_lantern",0.36,["light"]),
    e("event:platform_two_scene","location:platform_two",0.35,["scene"]),
    e("event:platform_two_scene","modifier:late_arrival",0.28,["timing"]),
    e("suspect:estranged_daughter","suspect:stationmaster",0.24,["witness"]),
    e("object:torn_ticket","suspect:stationmaster",0.24,["scene"]),
    e("object:telegram","suspect:stationmaster",0.32,["access"]),
    e("action:telegram_handled","suspect:stationmaster",0.34,["access"]),
    e("location:station_office","suspect:stationmaster",0.30,["access"]),
    # --- Paper trail cluster ---
    e("object:telegram","object:telegraph_form",0.34,["paper"]),
    e("location:station_office","object:ink_blotter",0.28,["desk"]),
    e("modifier:smudged","object:telegram",0.30,["ink"]),
    e("modifier:damp","time:fog_thickens",0.26,["weather"]),
    e("modifier:late_arrival","object:torn_ticket",0.30,["timing"]),
    e("modifier:late_arrival","suspect:estranged_daughter",0.24,["arrival"]),
    # --- Movement cluster ---
    e("action:footsteps_faded","location:platform_two",0.24,["trail"]),
    e("action:platform_crossed","location:platform_one",0.26,["movement"]),
    e("action:platform_crossed","location:platform_two",0.28,["movement"]),
    e("action:platform_crossed","object:coal_dust",0.24,["trace"]),
    e("location:goods_shed","object:coal_dust",0.36,["trace"]),
    e("location:platform_two","time:fog_thickens",0.24,["visibility"]),
    e("location:platform_one","time:last_arrival",0.22,["arrival"]),
    e("location:platform_one","time:first_departure",0.24,["departure"]),
    # --- Witness placement cluster ---
    e("action:signal_paused","witness:signalman",0.30,["log"]),
    e("location:signal_box","witness:signalman",0.30,["post"]),
    e("location:waiting_room","witness:porter",0.30,["post"]),
    e("location:platform_one","witness:ticket_clerk",0.30,["post"]),
    e("location:carriage_siding","witness:carriage_cleaner",0.26,["post"]),
    e("object:telegraph_form","witness:telegraph_operator",0.30,["handled"]),
    # --- Opening event cluster ---
    e("event:aldous_verne_discovered","location:platform_two",0.32,["scene"]),
    e("event:aldous_verne_discovered","time:between_trains",0.28,["time"]),
    e("event:aldous_verne_discovered","witness:porter",0.24,["found"]),
    e("emotion:paused_reply","witness:ticket_clerk",0.22,["interview"]),
    e("emotion:avoided_contact","suspect:night_porter",0.22,["interview"]),
    e("emotion:measured_breath","suspect:railway_clerk",0.26,["interview"]),
    e("emotion:steady_hands","suspect:stationmaster",0.24,["interview"]),
    e("emotion:closed_posture","suspect:travelling_broker",0.20,["interview"]),
    e("location:waiting_room","suspect:night_porter",0.26,["post"]),
    e("suspect:night_porter","witness:porter",0.24,["colleagues"]),
    e("location:platform_one","suspect:platform_guard",0.28,["post"]),
    e("object:platform_lantern","suspect:platform_guard",0.26,["equipment"]),
    e("location:waiting_room","suspect:travelling_broker",0.24,["presence"]),
    e("suspect:travelling_broker","witness:ticket_clerk",0.22,["ticket"]),
    # --- Context cluster ---
    e("location:thornfield_crossing","time:between_trains",0.24,["time"]),
    e("event:aldous_verne_discovered","location:thornfield_crossing",0.26,["scene"]),
    e("location:goods_shed","suspect:railway_clerk",0.24,["access"]),
    e("location:carriage_siding","object:coal_dust",0.24,["trace"]),
    e("modifier:paid_off","object:cash_tin",0.32,["payment"]),
    e("object:cipher_sheet","object:ink_blotter",0.28,["desk"]),
    e("emotion:fixed_stare","location:platform_two",0.22,["scene"]),
    e("emotion:measured_breath","object:station_key",0.22,["interview"]),
    e("emotion:steady_hands","object:station_key",0.24,["handover"]),
    e("emotion:paused_reply","time:first_departure",0.22,["interview"]),
    e("emotion:closed_posture","location:waiting_room",0.22,["post"]),
    e("location:station_office","modifier:unclaimed",0.24,["property"]),
    e("modifier:smudged","object:sealed_envelope",0.28,["ink"]),
    e("location:platform_two","modifier:lamps_dimmed",0.28,["light"]),
    e("modifier:ticket_punched","witness:ticket_clerk",0.26,["mark"]),
    e("action:ledger_checked","motive:ledger_exposure",0.30,["records"]),
    e("action:footsteps_faded","time:fog_thickens",0.24,["visibility"]),
    e("motive:debt_pressure","object:ledger_book",0.26,["note"]),
    e("object:coal_dust","witness:carriage_cleaner",0.24,["cleaning"]),
    e("location:station_office","witness:telegraph_operator",0.24,["post"]),
    e("emotion:avoided_contact","location:waiting_room",0.22,["interview"]),
    e("modifier:damp","object:sealed_envelope",0.24,["moisture"]),
]

NEW_EDGES = [
    # ---- Cluster A: Marco Laine false-lead (20) ----
    e("suspect:marco_laine","object:cipher_sheet",0.38,["evidence"]),
    e("suspect:marco_laine","object:satchel",0.32,["possession"]),
    e("suspect:marco_laine","location:waiting_room",0.30,["presence"]),
    e("suspect:marco_laine","witness:ticket_clerk",0.28,["witness"]),
    e("suspect:marco_laine","modifier:late_arrival",0.26,["timing"]),
    e("suspect:marco_laine","motive:third_party_link",0.34,["link"]),
    e("suspect:marco_laine","action:satchel_swapped",0.30,["exchange"]),
    e("suspect:marco_laine","suspect:laine_associate",0.40,["associate"]),
    e("suspect:laine_associate","object:sealed_envelope",0.28,["possession"]),
    e("suspect:laine_associate","location:waiting_room",0.26,["presence"]),
    e("suspect:laine_associate","witness:booking_clerk",0.24,["witness"]),
    e("suspect:marco_laine","object:torn_ticket",0.28,["travel"]),
    e("suspect:marco_laine","time:between_trains",0.24,["alibi"]),
    e("suspect:marco_laine","event:aldous_verne_discovered",0.26,["scene"]),
    e("suspect:marco_laine","emotion:fixed_stare",0.24,["behavior"]),
    e("suspect:marco_laine","modifier:unclaimed",0.24,["property"]),
    e("suspect:laine_associate","motive:third_party_link",0.30,["link"]),
    e("suspect:marco_laine","witness:luggage_porter",0.26,["witness"]),
    e("suspect:laine_associate","action:satchel_swapped",0.24,["exchange"]),
    e("suspect:marco_laine","object:telegram",0.28,["evidence"]),
    # ---- Cluster B: Elara Voss / accomplice (20) ----
    e("suspect:elara_voss","object:monogrammed_scarf",0.44,["possession"]),
    e("suspect:elara_voss","object:voss_gloves",0.40,["possession"]),
    e("suspect:elara_voss","action:log_altered",0.42,["evidence"]),
    e("suspect:elara_voss","event:elara_spotted",0.44,["presence"]),
    e("suspect:elara_voss","time:elara_arrival",0.38,["timing"]),
    e("suspect:elara_voss","action:rendezvous_kept",0.36,["meeting"]),
    e("suspect:elara_voss","emotion:shared_glance",0.34,["behavior"]),
    e("suspect:elara_voss","motive:elara_complicity",0.38,["motive"]),
    e("object:monogrammed_scarf","location:upper_corridor",0.34,["scene"]),
    e("object:voss_gloves","location:upper_corridor",0.30,["scene"]),
    e("event:voss_meeting","suspect:elara_voss",0.40,["scene"]),
    e("event:voss_meeting","suspect:renard_voss",0.38,["meeting"]),
    e("event:voss_meeting","location:waiting_room",0.30,["location"]),
    e("action:rendezvous_kept","event:voss_meeting",0.36,["meeting"]),
    e("modifier:two_visitors","event:voss_meeting",0.32,["count"]),
    e("modifier:two_visitors","location:waiting_room",0.28,["presence"]),
    e("object:monogrammed_scarf","witness:booking_clerk",0.28,["found"]),
    e("time:elara_arrival","witness:booking_clerk",0.30,["log"]),
    e("event:elara_spotted","witness:freight_handler",0.30,["sighting"]),
    e("suspect:elara_voss","emotion:guarded_proximity",0.32,["behavior"]),
    # ---- Cluster C: Office / location (20) ----
    e("location:stationmasters_office","action:key_duplicated",0.42,["access"]),
    e("location:stationmasters_office","object:access_register",0.40,["records"]),
    e("location:stationmasters_office","action:office_swept",0.38,["trace"]),
    e("location:stationmasters_office","location:upper_corridor",0.36,["adjacent"]),
    e("location:upper_corridor","action:corridor_cleared",0.34,["access"]),
    e("location:stationmasters_office","object:office_ledger",0.38,["contents"]),
    e("location:stationmasters_office","event:office_entry",0.42,["entry"]),
    e("object:access_register","action:log_altered",0.40,["alteration"]),
    e("object:key_impression","action:key_duplicated",0.42,["method"]),
    e("object:key_impression","object:station_key",0.36,["source"]),
    e("modifier:freshly_wiped","object:voss_gloves",0.34,["trace"]),
    e("modifier:freshly_wiped","location:stationmasters_office",0.30,["cleaning"]),
    e("object:door_wedge","location:stationmasters_office",0.28,["access"]),
    e("object:door_wedge","action:corridor_cleared",0.26,["prop"]),
    e("modifier:door_forced","location:records_room",0.30,["access"]),
    e("location:records_room","location:stationmasters_office",0.34,["adjacent"]),
    e("location:records_room","object:office_ledger",0.32,["contents"]),
    e("object:floor_plan","location:stationmasters_office",0.34,["blueprint"]),
    e("event:office_entry","time:office_window",0.38,["timing"]),
    e("time:corridor_empty","action:office_swept",0.32,["timing"]),
    # ---- Cluster D: Renard Voss / true suspect (20) ----
    e("suspect:renard_voss","object:falsified_entry",0.44,["evidence"]),
    e("suspect:renard_voss","event:cipher_exchange",0.40,["activity"]),
    e("suspect:renard_voss","action:seal_broken",0.38,["evidence"]),
    e("suspect:renard_voss","object:cipher_key",0.42,["tool"]),
    e("suspect:renard_voss","object:log_entry",0.40,["record"]),
    e("suspect:renard_voss","event:record_alteration",0.38,["activity"]),
    e("suspect:renard_voss","motive:record_destruction",0.36,["motive"]),
    e("suspect:renard_voss","action:rendezvous_kept",0.36,["meeting"]),
    e("object:cipher_key","object:cipher_sheet",0.42,["pair"]),
    e("event:cipher_exchange","object:cipher_key",0.38,["involved"]),
    e("event:cipher_exchange","object:cipher_sheet",0.36,["involved"]),
    e("action:seal_broken","object:sealed_envelope",0.38,["evidence"]),
    e("object:log_entry","object:ledger_book",0.36,["records"]),
    e("modifier:matched_handwriting","object:falsified_entry",0.38,["evidence"]),
    e("modifier:matched_handwriting","object:log_entry",0.34,["evidence"]),
    e("modifier:ink_matched","object:falsified_entry",0.36,["evidence"]),
    e("modifier:ink_matched","object:log_entry",0.32,["evidence"]),
    e("event:false_alibi","suspect:renard_voss",0.36,["alibi"]),
    e("event:false_alibi","suspect:elara_voss",0.34,["alibi"]),
    e("object:burned_note","motive:record_destruction",0.38,["evidence"]),
    # ---- Cluster E: Timing (15) ----
    e("time:office_window","time:between_trains",0.40,["overlap"]),
    e("time:office_window","time:signal_reset",0.34,["sequence"]),
    e("time:elara_arrival","time:between_trains",0.34,["overlap"]),
    e("time:before_discovery","time:between_trains",0.36,["sequence"]),
    e("time:corridor_empty","time:between_trains",0.32,["window"]),
    e("time:second_visit","time:office_window",0.28,["timing"]),
    e("time:ledger_opened","action:ledger_checked",0.34,["timing"]),
    e("time:ledger_opened","time:office_window",0.30,["overlap"]),
    e("time:before_discovery","event:aldous_verne_discovered",0.34,["timing"]),
    e("modifier:unscheduled","time:between_trains",0.28,["anomaly"]),
    e("modifier:unscheduled","action:office_entered",0.26,["anomaly"]),
    e("modifier:out_of_sequence","object:access_register",0.30,["anomaly"]),
    e("modifier:out_of_sequence","time:signal_reset",0.26,["timing"]),
    e("time:elara_arrival","event:aldous_verne_discovered",0.26,["timing"]),
    e("time:second_visit","event:voss_meeting",0.26,["timing"]),
    # ---- Cluster F: Cross-thread (29) ----
    e("suspect:vera_holt","object:receipt_book",0.34,["records"]),
    e("suspect:vera_holt","motive:debt_pressure",0.30,["link"]),
    e("suspect:vera_holt","witness:dispatch_runner",0.26,["associate"]),
    e("suspect:night_dispatch","action:message_relayed",0.36,["role"]),
    e("suspect:night_dispatch","object:dispatch_log",0.34,["records"]),
    e("suspect:night_dispatch","time:between_trains",0.28,["timing"]),
    e("suspect:office_keeper","object:station_key",0.36,["custody"]),
    e("suspect:office_keeper","location:stationmasters_office",0.34,["access"]),
    e("suspect:office_keeper","modifier:door_forced",0.30,["access"]),
    e("object:dispatch_log","time:signal_reset",0.28,["log"]),
    e("object:dispatch_log","event:window_between_trains",0.30,["timing"]),
    e("object:receipt_book","motive:fraud_concealment",0.36,["evidence"]),
    e("object:receipt_book","object:ledger_book",0.32,["records"]),
    e("object:account_slip","object:receipt_book",0.30,["records"]),
    e("object:account_slip","motive:debt_pressure",0.28,["debt"]),
    e("object:wax_seal_stamp","object:sealed_envelope",0.38,["tool"]),
    e("object:wax_seal_stamp","action:seal_broken",0.36,["tool"]),
    e("object:office_ledger","object:ledger_book",0.32,["records"]),
    e("object:office_ledger","motive:fraud_concealment",0.34,["records"]),
    e("motive:inheritance_blocked","suspect:renard_voss",0.34,["motive"]),
    e("motive:inheritance_blocked","motive:fraud_concealment",0.30,["link"]),
    e("motive:office_access_price","action:guard_bribed",0.32,["payment"]),
    e("motive:office_access_price","motive:fraud_concealment",0.28,["link"]),
    e("event:document_removal","location:stationmasters_office",0.36,["scene"]),
    e("event:document_removal","object:office_ledger",0.34,["target"]),
    e("event:ledger_burned","object:burned_note",0.40,["evidence"]),
    e("event:ledger_burned","object:office_ledger",0.32,["target"]),
    e("action:records_pulled","location:records_room",0.34,["scene"]),
    e("action:message_relayed","event:cipher_exchange",0.30,["chain"]),
    # ---- Low-degree fixes ----
    # accomplice:elara_voss invariant needs ≥ 2 edges (no inv-inv edges)
    e("accomplice:elara_voss","suspect:elara_voss",0.48,["identity"]),
    e("accomplice:elara_voss","action:log_altered",0.44,["evidence"]),
    e("accomplice:elara_voss","object:monogrammed_scarf",0.42,["possession"]),
    # witness:lamp_trimmer (0 edges)
    e("witness:lamp_trimmer","location:upper_corridor",0.26,["post"]),
    e("witness:lamp_trimmer","time:corridor_empty",0.22,["timing"]),
    # witness:freight_handler (1 edge → needs 1 more)
    e("witness:freight_handler","location:back_platform",0.24,["post"]),
    # witness:dispatch_runner (1 edge → needs 1 more)
    e("witness:dispatch_runner","object:dispatch_log",0.26,["records"]),
    # witness:coal_boy (0 edges)
    e("witness:coal_boy","location:goods_shed",0.24,["post"]),
    e("witness:coal_boy","object:coal_dust",0.22,["trace"]),
    # witness:night_cleaner (0 edges)
    e("witness:night_cleaner","location:upper_corridor",0.22,["post"]),
    e("witness:night_cleaner","time:before_discovery",0.20,["timing"]),
    # witness:platform_vendor (0 edges)
    e("witness:platform_vendor","location:waiting_room",0.22,["post"]),
    e("witness:platform_vendor","event:voss_meeting",0.22,["sighting"]),
    # witness:office_attendant (0 edges)
    e("witness:office_attendant","location:upper_corridor",0.26,["post"]),
    e("witness:office_attendant","time:office_window",0.24,["gap"]),
    # witness:stationmaster_aide (0 edges)
    e("witness:stationmaster_aide","location:stationmasters_office",0.26,["post"]),
    e("witness:stationmaster_aide","time:ledger_opened",0.22,["records"]),
    # witness:luggage_porter (1 edge → needs 1 more)
    e("witness:luggage_porter","location:waiting_room",0.22,["post"]),
    # location:dispatch_office (0 or 1 edge)
    e("location:dispatch_office","object:dispatch_log",0.30,["contents"]),
    e("location:dispatch_office","suspect:night_dispatch",0.30,["post"]),
    # location:back_platform (1 edge → needs 1 more)
    e("location:back_platform","object:coal_dust",0.24,["trace"]),
    # location:goods_office (0 edges)
    e("location:goods_office","object:receipt_book",0.28,["contents"]),
    e("location:goods_office","location:goods_shed",0.24,["adjacent"]),
    # object:floor_plan (1 edge → needs 1 more)
    e("object:floor_plan","location:upper_corridor",0.28,["blueprint"]),
    # motive:elara_complicity (1 edge → needs 1 more)
    e("motive:elara_complicity","object:monogrammed_scarf",0.30,["possession"]),
    # action:records_pulled (1 edge)
    e("action:records_pulled","object:office_ledger",0.32,["target"]),
    # emotion:shared_glance (1 edge)
    e("emotion:shared_glance","event:voss_meeting",0.28,["behavior"]),
    # emotion:feigned_surprise (0 edges)
    e("emotion:feigned_surprise","event:aldous_verne_discovered",0.26,["reaction"]),
    e("emotion:feigned_surprise","suspect:elara_voss",0.28,["behavior"]),
    # emotion:coordinated_calm (0 edges)
    e("emotion:coordinated_calm","suspect:renard_voss",0.26,["behavior"]),
    e("emotion:coordinated_calm","suspect:elara_voss",0.26,["behavior"]),
    # emotion:rehearsed_alibi (0 edges)
    e("emotion:rehearsed_alibi","event:false_alibi",0.30,["evidence"]),
    e("emotion:rehearsed_alibi","suspect:renard_voss",0.24,["behavior"]),
    # emotion:guarded_proximity (1 edge)
    e("emotion:guarded_proximity","event:voss_meeting",0.28,["behavior"]),
    # emotion:deflected_question (0 edges)
    e("emotion:deflected_question","witness:office_attendant",0.24,["interview"]),
    e("emotion:deflected_question","location:upper_corridor",0.22,["interview"]),
    # modifier:twice_signed (0 edges)
    e("modifier:twice_signed","object:access_register",0.32,["mark"]),
    e("modifier:twice_signed","object:falsified_entry",0.28,["mark"]),
    # modifier:recently_moved (0 edges)
    e("modifier:recently_moved","object:door_wedge",0.26,["placement"]),
    e("modifier:recently_moved","location:upper_corridor",0.24,["clue"]),
    # modifier:still_warm (0 edges)
    e("modifier:still_warm","object:burned_note",0.28,["trace"]),
    e("modifier:still_warm","location:stationmasters_office",0.26,["trace"]),
    # event:record_alteration (1 edge)
    e("event:record_alteration","object:office_ledger",0.34,["target"]),
]

all_edges = ORIG_EDGES + NEW_EDGES
print(f"Edge count: {len(all_edges)} (orig={len(ORIG_EDGES)} new={len(NEW_EDGES)})")

# Verify no duplicate pairs
seen = set()
dupes = []
for ed in all_edges:
    pair = tuple(sorted([ed["from"], ed["to"]]))
    if pair in seen:
        dupes.append(pair)
    seen.add(pair)
if dupes:
    print(f"WARNING: duplicate edge pairs: {dupes}")
else:
    print("No duplicate edge pairs ✓")

# ---------------------------------------------------------------------------
# EXPRESSIONS
# ---------------------------------------------------------------------------
EXPR = {
    # existing
    "suspect:renard_voss": "Renard Voss signed the station ledger in a tight hand. His coat hem is damp with platform mist.",
    "suspect:estranged_daughter": "A young woman waited at platform two with a small bouquet. The stems are wrapped in railway paper.",
    "suspect:stationmaster": "The stationmaster's name is on duty for the 11pm watch. His brass watch chain is tucked away.",
    "suspect:night_porter": "The night porter carries a brass key ring with three keys missing. His boots are clean of coal dust.",
    "suspect:platform_guard": "The platform guard's lantern hook is empty. His gloves are dry despite the fog.",
    "suspect:railway_clerk": "The railway clerk has fresh ink on two fingers. His cuffs are neatly folded back.",
    "suspect:travelling_broker": "A broker in a grey coat waited in the lounge. His ticket is first class and unused.",
    "location:thornfield_crossing": "Thornfield Crossing is a two-platform junction with a single office door. The clock above the gate shows 11:07.",
    "location:platform_two": "Platform two is lit by three lamps. The far bench is wet.",
    "location:platform_one": "Platform one faces the signal box. A luggage cart blocks the east end.",
    "location:station_office": "The station office has one desk and a locked cabinet. The window latch is unfastened.",
    "location:signal_box": "The signal box lever for the siding is down. A mug of cold tea sits by the panel.",
    "location:waiting_room": "The waiting room stove is cold. Two chairs are pulled close together.",
    "location:goods_shed": "The goods shed floor is dusted with coal. A crate lid is ajar.",
    "location:carriage_siding": "The carriage siding holds one empty carriage. The steps are damp.",
    "object:telegram": "A folded telegram lies on the office desk. The seal is broken and the paper is creased.",
    "object:cipher_sheet": "A cipher sheet is pinned beneath the blotter. Several columns are marked in pencil.",
    "object:satchel": "A brown satchel rests by the bench. The clasp bears the initials RV in brass.",
    "object:initialed_cufflink": "A cufflink with the letters RV is on the office floor. The hinge is bent.",
    "object:station_key": "A heavy station key sits in the drawer. Its tag reads OFFICE.",
    "object:ledger_book": "The ledger book is open to the accounts page. A line of figures is crossed through.",
    "object:sealed_envelope": "A sealed envelope is tucked inside the ledger. The wax is stamped with a crest.",
    "object:cash_tin": "A cash tin is half full of coins. The lid shows a fresh scrape.",
    "object:platform_lantern": "A platform lantern is unlit on the bench. The wick is trimmed short.",
    "object:torn_ticket": "A torn ticket half is pinned to the wall board. The date reads 12 November.",
    "object:telegraph_form": "A telegraph form lies in the waste bin. The message lines are blank.",
    "object:ink_blotter": "The ink blotter is damp on one corner. The blot has a sharp edge.",
    "object:coal_dust": "Coal dust is scattered on the platform steps. The pattern forms a clear shoe print.",
    "emotion:fixed_stare": "She did not look at the body. She looked at the track beyond.",
    "emotion:avoided_contact": "He answered without meeting the inspector's eyes. His gaze stayed on the timetable.",
    "emotion:measured_breath": "Her breathing slowed as questions were asked. The reply came after two beats.",
    "emotion:steady_hands": "His hands did not shake while handing over the keys. The ring did not jingle.",
    "emotion:paused_reply": "A pause came before the time was given. The answer was short and exact.",
    "emotion:closed_posture": "Arms were kept close to the body. The coat remained fastened indoors.",
    "modifier:initialed_RV": "The initials RV appear on two separate items. The lettering is the same style.",
    "modifier:paid_off": "Coins were found stacked in a neat column. The stack does not match the till count.",
    "modifier:unlocked": "The office latch turns without resistance. The lock shows no tool marks.",
    "modifier:misplaced": "An item rests where it would not be left in routine use. The placement blocks a clear path.",
    "modifier:unclaimed": "No name is written on the property tag. The string is newly tied.",
    "modifier:smudged": "A line of ink is smudged at the margin. The smear runs left to right.",
    "modifier:damp": "Moisture beads on the paper edge. The rest of the sheet is dry.",
    "modifier:late_arrival": "The register shows a signature after the last arrival time. The ink is fresh.",
    "modifier:lamps_dimmed": "The platform lamps burn lower than usual. The glass is clean.",
    "modifier:ticket_punched": "The ticket is punched twice in the same place. The second punch is deeper.",
    "action:office_entered": "The office door was used; the hinge oil is disturbed. A footprint marks the mat.",
    "action:telegram_handled": "The telegram is unfolded and refolded. A thumbprint sits near the code header.",
    "action:satchel_swapped": "Two satchels were present; one is missing. The strap buckle is left behind.",
    "action:ledger_checked": "The ledger has a new crease at the fraud column. The page corner is bent.",
    "action:platform_crossed": "Footprints cross from platform one to two. The prints are over fresh dust.",
    "action:signal_paused": "The signal lever was held at danger. The chalk mark is erased.",
    "action:guard_bribed": "A coin pouch was found under the guard's bench. The string is untied.",
    "action:door_left_open": "The office door was left ajar. The latch is not fully seated.",
    "action:footsteps_faded": "Footsteps lead from the bench into fog. The trail ends at the platform edge.",
    "time:between_trains": "The interval between the last and first trains is thirty-two minutes. No passenger traffic occurs in that span.",
    "time:last_arrival": "The last train arrived at 10:58. The platform cleared by 11:03.",
    "time:first_departure": "The first morning departure is listed at 11:30. The signal is set at 11:25.",
    "time:fog_thickens": "Fog thickened after the last arrival. Visibility reduced to ten yards.",
    "time:signal_reset": "The signal was reset at 11:24. The lever shows a fresh grease mark.",
    "motive:fraud_concealment": "The telegram contains proof of systematic fraud. The names are written in full.",
    "motive:third_party_link": "A third party of social standing is named in the cipher. The name is underlined.",
    "motive:partnership_ruin": "The partnership accounts show a final deficit. The balance would end the firm.",
    "motive:ledger_exposure": "A ledger column titled dividends does not match the receipts. The discrepancy is dated.",
    "motive:debt_pressure": "A promissory note is clipped inside the ledger. The due date is past.",
    "motive:sealed_proof": "The sealed proof is addressed to authorities. The seal is intact but warm.",
    "witness:ticket_clerk": "The ticket clerk recorded a late punch. The log notes platform two.",
    "witness:porter": "The porter moved luggage at 11:05. His cart track ends by the bench.",
    "witness:signalman": "The signalman logged a pause at 11:12. The entry is initialed.",
    "witness:carriage_cleaner": "The carriage cleaner wiped the siding steps at 11:15. The rag is still damp.",
    "witness:telegraph_operator": "The telegraph operator handled a blank form at 11:02. The tray is empty now.",
    "event:aldous_verne_discovered": "Aldous Verne was found on platform two. His watch stopped at 11:16.",
    "event:platform_two_scene": "Platform two was cleared after the last arrival. Only one suitcase remained.",
    "event:window_between_trains": "The thirty-two minute window between trains is recorded. The office was unoccupied then.",
    # new invariants
    "location:stationmasters_office": "The stationmaster's private office overlooks the main hall. The inner lock is a different pattern to the outer door.",
    "accomplice:elara_voss": "Elara Voss — her monogram appears on the scarf left in the upper corridor. The access register shows a second hand in the entry for 11:10.",
    # new non-invariant
    "suspect:marco_laine": "Marco Laine arrived on the 10:45 from the city. His briefcase has a broken latch and a new lock.",
    "suspect:vera_holt": "Vera Holt spoke briefly at the booking window. Her gloves are the same cut as those found in the office.",
    "suspect:elara_voss": "A woman matching the description of Elara Voss was seen near the upper corridor at 11:10. She left no name at the desk.",
    "suspect:night_dispatch": "The night dispatch clerk is responsible for all telegrams sent after 10:00. His log shows an erasure.",
    "suspect:office_keeper": "The office keeper holds a duplicate key to the stationmaster's room. His shift ended at 11:00.",
    "suspect:laine_associate": "A man travelled with Marco Laine. He waited at the far end of the waiting room and did not approach the desk.",
    "witness:booking_clerk": "The booking clerk noted two visitors to the upper corridor between 11:05 and 11:20. Neither signed the register.",
    "witness:lamp_trimmer": "The lamp trimmer was working the upper corridor at 11:08. He heard a door close on the left side.",
    "witness:freight_handler": "The freight handler unloaded a crate at 11:12. He saw a woman pass through the goods archway.",
    "witness:dispatch_runner": "The dispatch runner carried a message to the east platform at 11:14. He passed no one on the stairs.",
    "witness:coal_boy": "The coal boy refilled the goods shed scuttle at 11:00. The corridor door was already open.",
    "witness:night_cleaner": "The night cleaner mopped the corridor floor at 10:55. The upper corridor was clear at that time.",
    "witness:platform_vendor": "The platform vendor saw two people exchange something near the waiting room door. Neither looked at him.",
    "witness:office_attendant": "The office attendant was positioned near the stationmaster's corridor from 11:00. He left for five minutes at 11:10.",
    "witness:stationmaster_aide": "The stationmaster's aide filed papers until 10:58. He last checked the inner office at 10:50.",
    "witness:luggage_porter": "The luggage porter moved two bags at 11:03. One bag had no label.",
    "location:upper_corridor": "The upper corridor connects the main hall to the stationmaster's office. The floor shows two sets of recent prints.",
    "location:records_room": "The records room adjoins the stationmaster's office. The filing cabinet nearest the window has a forced latch.",
    "location:dispatch_office": "The dispatch office handles overnight telegraph traffic. The outgoing tray is empty despite the log showing three messages.",
    "location:back_platform": "The back platform is used for freight and unscheduled stops. It is not visible from the main hall.",
    "location:lamp_room": "The lamp room stores spare lanterns and wick oil. The door was left unlocked after the last trim.",
    "location:goods_office": "The goods office records all incoming and outgoing freight. The ledger page for 12 November has a torn corner.",
    "object:voss_gloves": "A pair of fine leather gloves were found near the stationmaster's office door. The right glove is freshly wiped.",
    "object:wax_seal_stamp": "A wax seal stamp with a crest lies in the desk drawer. The stamp does not match the crest on the official seal.",
    "object:access_register": "The access register records every visitor to the upper corridor. The entry for 11:10 is crossed out in a different ink.",
    "object:falsified_entry": "A falsified entry appears in the accounts book. The handwriting matches two other documents in the office.",
    "object:receipt_book": "The receipt book shows a payment made on 10 November. The payee line is left blank.",
    "object:cipher_key": "A small card with a cipher key is folded inside the satchel lining. The key matches the columns on the cipher sheet.",
    "object:dispatch_log": "The dispatch log records three outgoing telegrams between 11:00 and 11:30. The destinations are not listed.",
    "object:floor_plan": "A hand-drawn floor plan of the station is pinned inside the office cupboard. The stationmaster's office is circled.",
    "object:burned_note": "A partially burned note was found in the office grate. The surviving line reads: confirm before the first train.",
    "object:office_ledger": "The stationmaster's private ledger is kept in a locked drawer. Three pages have been removed.",
    "object:monogrammed_scarf": "A wool scarf monogrammed EV was left on the upper corridor bench. The tag shows it was purchased in the city.",
    "object:log_entry": "A log entry for 11:08 records access to the records room. The initials do not match any known staff member.",
    "object:door_wedge": "A wooden door wedge was found propped against the corridor fire door. It is not standard station equipment.",
    "object:key_impression": "A wax impression of a key was found wrapped in paper in the desk. The profile matches the office key.",
    "object:account_slip": "A loose account slip tucked between pages shows a sum that does not appear in the main ledger.",
    "motive:inheritance_blocked": "A legal notice in the sealed envelope records that a substantial inheritance was contested and frozen. The named claimant is Renard Voss.",
    "motive:elara_complicity": "Elara Voss stood to gain from the destruction of the ledger records. Her name appears on a secondary account.",
    "motive:office_access_price": "A cash payment was made to secure access to the stationmaster's office on the night in question. The amount matches the cash tin deficit.",
    "motive:record_destruction": "The missing ledger pages contained the only dated record of the fraudulent transfers. Their removal destroys the evidence trail.",
    "action:log_altered": "The access log has been altered. The original entry time has been overwritten in a finer hand.",
    "action:office_swept": "The stationmaster's office floor was swept after the last staff departure. A single fine hair remains near the door frame.",
    "action:key_duplicated": "The wax impression confirms a key was duplicated before the night of the incident. The locksmith's tool mark is on the original key.",
    "action:message_relayed": "A verbal message was relayed through the dispatch runner at 11:14. The content was not recorded.",
    "action:records_pulled": "Three pages were removed from the records room filing cabinet. The gaps in the sequence are dated to the fraud period.",
    "action:corridor_cleared": "The upper corridor was cleared of staff between 11:08 and 11:20. The office attendant's log shows a five-minute absence.",
    "action:seal_broken": "The official seal on the envelope has been broken and replaced. The second application of wax is fractionally thinner.",
    "action:rendezvous_kept": "A planned meeting was kept in the waiting room at 11:05. Two chairs were moved together and not returned.",
    "emotion:shared_glance": "Two witnesses described a silent exchange of glances between a man and a woman near the waiting room door.",
    "emotion:feigned_surprise": "The reaction on hearing of the discovery appeared rehearsed. The expression did not reach the eyes.",
    "emotion:coordinated_calm": "Both individuals questioned showed the same measured composure. Neither asked about the victim.",
    "emotion:rehearsed_alibi": "The account of movements given was precise to the minute. The same phrase appeared in both statements.",
    "emotion:guarded_proximity": "The woman remained within arm's reach of the man throughout the wait. She did not speak to anyone else.",
    "emotion:deflected_question": "Direct questions about the upper corridor were redirected to a secondary topic. The deflection was practised.",
    "modifier:freshly_wiped": "The surface near the office door has been freshly wiped. The cleaning cloth was found discarded in the lamp room.",
    "modifier:twice_signed": "The register page bears two signatures in the same hand. The second was added after the first dried.",
    "modifier:out_of_sequence": "The access log entries for 11:05 to 11:20 are out of sequence with the rest of the register.",
    "modifier:recently_moved": "The chair nearest the corridor door had been recently moved. The floor marks do not match its current position.",
    "modifier:matched_handwriting": "Two documents in the office were written by the same hand. The letterforms match those on the satchel label.",
    "modifier:still_warm": "The office grate still held warmth at 11:30 despite the fire being unlit for hours. Something had been burned recently.",
    "modifier:door_forced": "The records room door shows a fresh mark consistent with a thin tool. The lock was not damaged but the frame was stressed.",
    "modifier:two_visitors": "Two sets of footprints lead to the upper corridor door. Only one set continues into the records room.",
    "modifier:ink_matched": "The ink on the falsified entry matches the ink on the satchel label. Both are a city-made variety not used by station staff.",
    "modifier:unscheduled": "The second visit to the office was not entered in any official log. No authorisation is recorded.",
    "time:office_window": "The stationmaster's office was unwatched between 11:08 and 11:20. This window corresponds exactly to the signal pause.",
    "time:elara_arrival": "Elara Voss arrived at Thornfield Crossing on the 10:45 service. Her presence was not announced.",
    "time:before_discovery": "The period between 11:10 and 11:16 is unaccounted for in the witness logs. Aldous Verne was found at 11:16.",
    "time:corridor_empty": "The upper corridor had no staff between 11:08 and 11:20. The lamp trimmer left at 11:08 and the attendant at 11:10.",
    "time:ledger_opened": "The stationmaster's ledger was last opened at 11:05 according to the desk log. The desk was found disturbed.",
    "time:second_visit": "A second visit to the waiting room was recorded at 11:18. The same two chairs had been moved again.",
    "event:voss_meeting": "Renard and Elara Voss met in the waiting room at 11:05. A witness observed them exchange a small folded item.",
    "event:office_entry": "The stationmaster's office was entered between 11:10 and 11:20. The key used was not the official station key.",
    "event:record_alteration": "Records in the stationmaster's office were altered during the unwatched window. Three page entries were changed.",
    "event:cipher_exchange": "The cipher key changed hands at the station between the 10:45 arrival and 11:05. The exchange point was the waiting room.",
    "event:document_removal": "Documents were removed from the records room during the incident window. The removal was systematic, not opportunistic.",
    "event:elara_spotted": "Elara Voss was observed near the upper corridor at 11:10 by the freight handler. She was alone.",
    "event:ledger_burned": "An attempt was made to burn several pages from the office ledger in the grate. The attempt was interrupted.",
    "event:false_alibi": "A coordinated false alibi was established placing both Renard and Elara Voss in the waiting room until 11:20. The account cannot be independently confirmed.",
    "event:evidence_planted": "A piece of evidence bearing Marco Laine's name was found in an unusual position. Its placement appears deliberate.",
}

# ---------------------------------------------------------------------------
# ASSEMBLE
# ---------------------------------------------------------------------------
case = {
    "case_id": "amber_cipher_L",
    "title": "The Amber Cipher: The Long Case",
    "convergence_shape": "late_break",
    "difficulty": "hard",
    "min_turns": 15,
    "max_turns": 28,
    "convergence_rate": 0.38,
    "convergence_threshold": 0.75,
    "opening_token_ids": [
        "location:thornfield_crossing",
        "event:aldous_verne_discovered",
        "time:between_trains"
    ],
    "invariant_token_ids": [
        "suspect:renard_voss",
        "event:window_between_trains",
        "motive:fraud_concealment",
        "location:stationmasters_office",
        "accomplice:elara_voss"
    ],
    "tokens": all_tokens,
    "graph": {
        "nodes": all_ids,
        "edges": all_edges,
    },
    "attractor": {
        "dimensions": [
            {"id": "killer_dimension",      "invariant_token": "suspect:renard_voss",           "threshold": 0.75, "label": "who acted"},
            {"id": "mechanism_dimension",   "invariant_token": "event:window_between_trains",   "threshold": 0.75, "label": "how it was done"},
            {"id": "motive_dimension",      "invariant_token": "motive:fraud_concealment",      "threshold": 0.75, "label": "why it happened"},
            {"id": "location_dimension",    "invariant_token": "location:stationmasters_office","threshold": 0.75, "label": "where it happened"},
            {"id": "accomplice_dimension",  "invariant_token": "accomplice:elara_voss",         "threshold": 0.75, "label": "who assisted"},
        ],
        "convergence_display": "minimum"
    },
    "expressions": EXPR,
}

# Verify all token IDs have an expression
missing_expr = [t["id"] for t in all_tokens if t["id"] not in EXPR]
if missing_expr:
    print(f"WARNING: missing expressions for: {missing_expr}")
else:
    print("All expressions present ✓")

OUT.write_text(json.dumps(case, indent=2))
print(f"Written: {OUT}  ({OUT.stat().st_size:,} bytes)")
