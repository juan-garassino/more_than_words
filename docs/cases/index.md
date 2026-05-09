# Living Tales — Case Index

Twenty-four cases across mystery, naval, and adventure formats. One engine.

Each case is a self-contained DLC: a vocabulary of tokens, an affinity graph, a convergence spec, and a trained transformer policy.

---

## Mystery Cases (01–20)

| # | Title | Case ID | Period | Setting | Difficulty | Vocab | Status |
|---|---|---|---|---|---|---|---|
| 01 | The Amber Cipher | `amber_cipher` | 1887 | Railway junction, Essex | medium | 72 | trained, proof passed |
| 02 | The Venetian Mirror | `venetian_mirror` | 1931 | Palazzo, Venice carnival | hard | 72 | draft JSON |
| 03 | Fog Over Brussels | `fog_over_brussels` | 1961 | Belgian embassy | hard | 72 | draft JSON |
| 04 | The Hollow Season | `hollow_season` | 1907 | Edwardian country house | medium | 72 | draft JSON |
| 05 | The Resonance Test | `resonance_test` | 1974 | Music conservatory, London | easy | 72 | draft JSON |
| 06 | The Tidal Interval | `tidal_interval` | Present | Island research station | medium | 72 | draft JSON |
| 07 | The Third Signature | `third_signature` | 1935 | London literary club | hard | 72 | draft JSON |
| 08 | The Sulphur Line | `sulphur_line` | 1889 | Victorian chemical works | medium | 72 | draft JSON |
| 09 | The Orchard at Dusk | `orchard_at_dusk` | 1903 | Rural England, harvest | easy | 72 | draft JSON |
| 10 | The Attended Hour | `attended_hour` | Present | Hospital cardiac ward | hard | 72 | draft JSON |
| 11 | The Winter Station | `winter_station` | 1912 | Antarctic research depot | medium | 72 | narrative only |
| 12 | The Monsoon Ledger | `monsoon_ledger` | 1905 | Calcutta, Bengal Partition | medium | 72 | narrative only |
| 13 | The Observatory Clock | `observatory_clock` | 1900 | Paris Observatory | medium | 72 | narrative only |
| 14 | The Endgame | `endgame` | 1972 | Reykjavik chess championship | hard | 72 | narrative only |
| 15 | The Amber Silence | `amber_silence` | 1943 | Occupied Normandy | hard | 72 | narrative only |
| 16 | The Signal Fire | `signal_fire` | 1943 | Pacific island, WWII | hard | 72 | narrative only |
| 17 | The Covenant Garden | `covenant_garden` | 1349 | Yorkshire monastery | medium | 72 | narrative only |
| 18 | The Mountain Exchange | `mountain_exchange` | 1938 | Swiss Alps, pre-war | hard | 72 | narrative only |
| 19 | The Instrument Landing | `instrument_landing` | 1954 | Post-war London | medium | 72 | narrative only |
| 20 | The Burning Glass | `burning_glass` | 1909 | Istanbul, Young Turk era | medium | 72 | narrative only |

## Naval Case (21)

| # | Title | Case ID | Period | Setting | Difficulty |
|---|---|---|---|---|---|
| 21 | The Dead Calm | `dead_calm` | 1698 | Pirate brigantine, Caribbean | hard |

*Closed-environment case: all tokens belong to people, objects, or events on a single ship during a four-day dead calm. No outside. No elsewhere.*

## Adventure Cases (A01–A03)

Adventure cases converge toward a **chosen state** rather than a fixed truth. The player's choices determine what becomes possible. The field records them.

| # | Title | Case ID | Period | Protagonist | Basins |
|---|---|---|---|---|---|
| A01 | The Thirteenth Tide | `thirteenth_tide` | 1697 | Sera Vane, cartographer's daughter | 4 |
| A02 | The Glass Cartographer | `glass_cartographer` | 1627 | Lena Faber, glassmaker's daughter | 4 |
| A03 | The Iron Cartridge | `iron_cartridge` | 1876 | Elias Drum, interpreter | 4 |

---

## Invariants — mystery and naval cases

| Case | Killer | Mechanism | Motive |
|---|---|---|---|
| amber_cipher | `suspect:renard_voss` | `event:window_between_trains` | `motive:fraud_concealment` |
| venetian_mirror | `suspect:countess_morvaine` | `event:balcony_fall` | `motive:forgery_operation` |
| fog_over_brussels | `suspect:cultural_attache` | `event:champagne_compound` | `motive:double_agent_exposure` |
| hollow_season | `suspect:edmund_carrow` | `event:sleeping_brandy` | `motive:inheritance_entailed` |
| resonance_test | `suspect:marta_solis` | `event:stairwell_fall` | `motive:stolen_composition` |
| tidal_interval | `suspect:piers_dunne` | `event:tidal_window` | `motive:fishing_economy` |
| third_signature | `suspect:agnes_vail` | `event:evening_port` | `motive:1921_scandal` |
| sulphur_line | `suspect:silas_oake` | `event:canal_path` | `motive:safety_falsification` |
| orchard_at_dusk | `suspect:thomas_crale` | `event:altered_path` | `motive:boundary_fraud` |
| attended_hour | `suspect:callum_dray` | `event:dosage_interaction` | `motive:negligence_testimony` |
| winter_station | `suspect:dr_marsh` | `event:ice_crossing` | `motive:mining_concealment` |
| monsoon_ledger | `suspect:gerald_peel` | `event:quinine_substitution` | `motive:uprising_concealment` |
| observatory_clock | `suspect:professor_serin` | `event:ladder_failure` | `motive:survey_espionage` |
| endgame | `suspect:nikolai_vronsky` | `event:mineral_water_compound` | `motive:defection_intelligence` |
| amber_silence | `suspect:henri_dumont` | `event:bell_tower_door` | `motive:milice_protection` |
| signal_fire | `suspect:captain_hayes` | `event:radio_modification` | `motive:deception_exposure` |
| covenant_garden | `suspect:brother_anselm` | `event:foxglove_tisane` | `motive:library_preservation` |
| mountain_exchange | `suspect:brand` | `event:safety_wire` | `motive:double_agent_protection` |
| instrument_landing | `suspect:arthur_devane` | `event:altimeter_modification` | `motive:component_trafficking` |
| burning_glass | `suspect:petrakis` | `event:warehouse_fire` | `motive:naval_intelligence` |
| dead_calm | `suspect:aldric_noe` | `event:surgical_instrument_wound` | `motive:pattern_exposure` |

---

## Adventure final triads

| Case | Token 1 | Token 2 | Token 3 |
|---|---|---|---|
| thirteenth_tide | `object:survey_notes` | `object:castillo_document` | `event:tomas_negotiation` |
| glass_cartographer | `ability:glass_reading` | `object:complete_route` | `location:basement_door` |
| iron_cartridge | `object:affidavit` | `character:ida_crane` | `event:morning_edition` |

---

## Cross-case connections

| Connection | Cases |
|---|---|
| The cardiac accelerant (tasteless, period-undetectable) appears in both Brussels and the Aldgate Club | 03, 07 |
| Renard Voss (killer, amber_cipher) has a namesake: Helena Voss, ambassador's wife, Brussels | 01, 03 |
| Maren Solís (Dead Calm captain) shares a surname with Marta Solís (Resonance Test killer) | 05, 21 |
| James Okafor (Case 05) / Dr. Sarah Okafor (Case 06) — explicitly "no relation" | 05, 06 |
| "Aldgate" names both the conservatory (Case 05) and the literary club (Case 07) | 05, 07 |
| The compound family in Cases 03 and 07 spans twenty-four years and two countries | 03, 07 |
| "Thornfield" appears as a village name in Case 09 | 09 |
| The Thirteenth Tide (A01) takes place one year before The Dead Calm (21); both are Caribbean, 1697/1698 | A01, 21 |
| Cases 10 and 16 both close without clean moral resolution — the killer was right about something | 10, 16 |

---

## Structure patterns

The mechanism is always a physical constraint. A weakened structure, a compound in a glass, a time window, a path in the dark. The killer never improvises. They use what the world already provides, shaped slightly.

The motive is always buried. The surface motive is visible by turn 8. The buried motive is older than the crime — from 1884, from 1916, from a decision made before the victim existed as a threat.

The red herring is always sympathetic. None of them killed anyone. All of them carry enough weight to sustain suspicion through half the game.

Every case has a bridge token. The satchel, the gate, the seven minutes at the club door, the three-am chart note. The game is always about finding that token. Everything else is the field warming around it.

Adventure cases differ in one essential way: the player is an actor, not an observer. Their choices determine what becomes possible. The field records who they chose to be while getting there.

---

## Files

- Narrative specifications: `docs/cases/NN_case_id.md`
- Case JSON (vocabulary + graph): `cases/case_id.json`
- Direct pixel-art background workspaces: `art/case_id/direct_pixel_art_v1/`
- Packed spec (after s02): `living_tales/trainer/cases/case_id/`
- Trained transformer (after s04): `living_tales/trainer/outputs/case_id/policy.pt`
