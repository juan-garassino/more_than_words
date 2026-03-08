# Case 19 — The Instrument Landing
## London · Croydon Aerodrome · December · 1954

| | |
|---|---|
| **Case ID** | `instrument_landing` |
| **Difficulty** | medium |
| **Vocab** | 72 tokens |
| **Killer** | `suspect:arthur_devane` |
| **Mechanism** | `event:altimeter_modification` |
| **Motive** | `motive:component_trafficking` |
| **Red herring** | Raymond Caldwell — aircraft owner, property developer, who increased his aviation insurance substantially two weeks before Fenn's scheduled inspection; the timing appears deliberate |
| **False plateau** | Caldwell path fills `killer_dim` and `motive_dim` (insurance fraud, financial exposure); `mechanism_dim` breaks it — altimeter modification requires avionics knowledge; an OBJECT token places Devane alone on the aircraft the night before Fenn's scheduled inspection |
| **Bridge token** | The component transaction ledger — a second set of books in Devane's workshop's locked cabinet; carries the purchasing agent's name, component serial numbers, and dates that overlap with documented German civil aviation accidents |

---

Gilbert Fenn is forty-four, an Air Ministry inspector with eighteen years of service and a reputation for precision that his colleagues describe as admirable and his subjects describe as inconvenient. He arrives at Croydon on the morning of December 14th for a scheduled airworthiness review of three aircraft operated by Meridian Aviation. He has done this twice before at other operators. He expects a two-day visit.

He is dead by afternoon.

The aircraft he is travelling in — a de Havilland Dove, G-AMRV, borrowed from a colleague at the Ministry for the cross-London transit — makes its approach to Croydon in a December fog that has settled at roughly two hundred feet. The approach is otherwise routine. At fifty feet by the altimeter, the aircraft is still well above the runway threshold. At zero feet by the altimeter, the aircraft impacts the ground. The altimeter reads fifty feet higher than actual altitude throughout the descent.

The fog and the wreckage together are sufficient to obscure the modification. The instrument is destroyed in the impact. The official investigation finds no evidence of tampering. It records calibration drift as the probable contributing factor in adverse conditions.

Arthur Devane reads the investigation report six weeks later and files it without annotation.

---

## Devane

He is fifty-eight, qualified as an aircraft engineer in 1921, and has run Meridian Aviation's maintenance operation since the company's founding in 1948. He knows every instrument in his shop. He knows the Smiths Industries altimeter series — which is fitted to G-AMRV — from its calibration procedure and its failure modes. He knows that a calibration offset of fifty feet, introduced by adjusting the static pressure reference, is indistinguishable from natural drift when the instrument is new enough and the offset small enough. He knows it because he has held the instrument in his hands and adjusted it and held it in his hands again.

He worked alone on G-AMRV from nine-thirty to eleven-fifteen on the evening of December 13th. The aerodrome has a night security man, Hedges, who made his round at ten-fifteen and noted a light in the Meridian hangar. He did not check inside. This is a WITNESS token at MID phase. Hedges's log entry is the moment the investigation changes direction.

Devane's surface motive, when it surfaces, is straightforward: Fenn's review was going to revoke airworthiness certificates on three of Meridian's aircraft. The revocations would have closed the company. Six employees, two contracts, a lease on the hangar running through 1957 — all of it would have ended on Fenn's report. This is a real motive. It is not the buried one.

---

## The Component Chain

In 1950, Devane began selling surplus RAF components to a purchasing agent named Voss. Not Renard Voss — the Voss family is large — but a materials broker named Heinrich Voss, operating through a Rotterdam trading company, whose clients Devane understood to be European civil aviation operators rebuilding their postwar fleets from available stock. This was legal in general terms and not unusual in the early 1950s, when surplus RAF material was moving through dozens of channels into civilian use.

What Devane chose not to investigate was the condition provenance of the components he was selling. Some of them had come from accident-damaged aircraft that had passed through Meridian's workshop for assessment. The RAF's disposal procedure required that accident-damaged components be certified as unserviceable before sale. Devane was certifying some of them as serviceable. The difference between a serviceable and an unserviceable certification was, in several cases, his own judgment, applied alone, in the workshop at night.

Five components sold to Voss between 1950 and 1953 were subsequently installed in aircraft operated by a West German charter carrier. Three of those aircraft experienced mechanical incidents. Two of those incidents resulted in crashes. Eleven people died across the two crashes. The German investigation reports named specific component failures. The serial numbers in those reports match serial numbers in Devane's transaction records.

Fenn's inspection, beginning as a routine airworthiness review, had reached Meridian's component provenance. He had requested serial number records for components disposed of in the past five years. He had a methodology for cross-referencing disposal records against Air Safety Board incident reports. He had used it before. He was going to use it on December 15th.

---

## Caldwell

Raymond Caldwell is forty-seven and owns four aircraft, G-AMRV among them, as instruments of a property development operation that uses private aviation for site inspection travel. He is not a pilot. He employs pilots through Meridian. He is good at accumulating assets and careful about insuring them.

On November 28th, two weeks before Fenn's inspection, Caldwell increased his aviation insurance from a standard package to a comprehensive policy with accident death benefit. He did this because his broker had suggested it in October and he had delayed until November. He has documentation confirming this sequence. The documentation exists and is genuine.

He has no knowledge of avionics. He has never worked on an aircraft. He is not in the maintenance logs for any of his four aircraft. He has no reason to know Fenn was scheduled, because Meridian handled the inspection liaison and did not inform owners of routine reviews. He learned of Fenn's death from the aerodrome manager's telephone call.

The insurance timing is real and coincidental. The player who follows it is not wrong to notice it. The `mechanism_dim` is what resolves the path: a property developer with no avionics background, no access to the hangar at night, no knowledge of the inspection schedule. Caldwell's path collapses on the night of December 13th. Hedges's witness token places Devane in the hangar. The workshop's tool cabinet — an OBJECT token at LATE phase — places the altimeter calibration tools at Devane's hand.

---

## The Second Ledger

Devane keeps two sets of books. The first is the standard Meridian maintenance ledger: component receipts, service records, inspection logs. This is in the workshop's main cabinet and available to any inspector. The second is a bound accounts book, buff coloured, kept in a locked drawer in the inner workshop. It records the Voss transactions.

The second ledger is a LATE phase OBJECT token. Its surface expression is: *a buff accounts book, locked drawer, inner workshop.* The drawer requires a key. The key is on Devane's workshop key ring. The key ring is a MID phase OBJECT token, placed through the investigation of the night of December 13th — Hedges's log identifies the hangar light; the hangar light places Devane at the workshop; the workshop investigation reaches the inner room; the inner room contains the locked drawer.

The second ledger lists thirty-seven transactions between October 1950 and March 1953. Against eleven of the entries, Devane has written a small mark — a pencilled X — whose meaning is not stated. Cross-referencing the serial numbers against the Air Safety Board incident reports is a LATE phase ANALYSIS token: the investigation must have assembled both the ledger and the incident report tokens for the cross-reference to be possible. When both are placed, `motive_dim` shifts from the surface reading (company closure, certificate revocation) to the buried reading (eleven deaths, provenance falsification, a chain of evidence that Fenn was about to complete).

The pencilled X entries are the components that failed.

---

## Design notes

The altimeter modification is the series' most technically specific mechanism. The Smiths Industries barometric altimeter, fitted as standard to the de Havilland Dove series, uses a static pressure reference that can be offset by adjusting the reference port connection. The adjustment is small, reversible, and leaves no visible trace on the instrument casing. This level of specificity should be reflected in the mechanism token cluster: low temperature (certain), high weight on `mechanism_dim`, with technical vocabulary that distinguishes it clearly from anything a non-specialist could have done.

The postwar reconstruction context — surplus components, rebuilding fleets, available stock moving through informal channels — is the case's distinctive historical texture. The token field should carry period atmosphere: austerity London, the aerodrome in December fog, the specific grey of 1954 institutional buildings. These are SETTING tokens that do not load onto the invariants but give the field its particular weight.

Caldwell is the most formally innocent red herring in the series. He did nothing wrong. His insurance increase was legitimate. His property development operation is real. His ignorance of the inspection process is genuine. He is a red herring not because he looks guilty through behaviour but because one circumstantial coincidence — the insurance timing — will sustain the player's attention for six or seven turns before the mechanism breaks it. This is a deliberate design: the player follows Caldwell not because he behaves suspiciously but because one piece of evidence looks too neat. It is not neat. It is coincidence. The game has a few of these. The player must learn to test mechanism against motive and not reverse.

The German accident reports are the case's moral weight. Eleven people. Two crashes. Components that Devane certified alone, at night, in a workshop, applying a judgment he knew was optimistic. He is not a man who planned eleven deaths. He is a man who made a calculation about salvageable components and was wrong and then continued making the calculation. Fenn was not going to prove he planned anything. He was going to prove what Devane sold, and when, and what happened next.
