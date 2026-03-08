# Case 08 — The Sulphur Line
## Victorian Industrial City · Chemical Works · Winter · 1889

| | |
|---|---|
| **Case ID** | `sulphur_line` |
| **Difficulty** | medium |
| **Vocab** | 72 tokens |
| **Killer** | `suspect:silas_oake` |
| **Mechanism** | `event:canal_path` |
| **Motive** | `motive:safety_falsification` |
| **Red herring** | James Whitlow (compensated witness) — grievance, proximity, unsigned document, seen near the works twice |
| **False plateau** | `killer_dim` stalls between Oake and Whitlow; `motive_dim` cannot close on Whitlow because his grievance is the opposite of suppression; `mechanism_dim` resolves only via Whitlow's LATE phase testimony about September 1886 |
| **Bridge token** | The three-day gap in the accident log — not the gap itself but the shadow of it: a reference to a note that should have followed an earlier entry and did not |

---

The Hartwick Chemical Works occupies twelve acres on the eastern edge of a northern industrial city whose name the case does not specify. It produces sulphuric acid, bleaching compounds, and a proprietary chemical used in textile processing that has made its owner moderately wealthy and its workers moderately ill. The canal that runs along its southern edge is the colour of weak tea and smells of something that is not quite sulphur and not quite anything else.

The works manager is Silas Oake. He has worked here for twenty-three years. His safety record — formally — is immaculate.

---

## The Inspector

Henry Gould is forty-four, a factory inspector employed by the Board of Trade, meticulous in a way that some find admirable and most find inconvenient. He arrives on a Monday in February for a routine inspection. He has done this before — he inspected the Hartwick works in 1882 and found minor violations that were corrected within the statutory period. He expects a similar visit.

He does not find a similar visit.

On his second day he requests the maintenance logs for the past five years. He finds that the valve inspection records for the third processing line have been revised. Not forged — revised. Earlier entries have been overwritten in a slightly different ink with slightly different handwriting. Someone has gone back and corrected the record. Gould has spent twenty years reading maintenance logs. He knows the difference between a correction made at the time and a correction made later.

He also finds a three-day gap in the accident log from September 1886.

He spends his evenings walking the canal path while he thinks. He walks it every evening after his site visits. Oake knows this because he has been watching.

On Thursday evening Gould does not complete his walk.

---

## The September Incident

In September 1886 a valve on the third processing line failed during the night shift. The failure released a concentrated compound into the enclosed workspace. One worker — a man named Thomas Birch — was exposed at length before the ventilation could be opened. He died in hospital four days later. Oake recorded the incident as a maintenance failure and a natural workplace hazard. He paid Birch's widow a sum of money and had her sign a document she did not fully understand. He paid the two workers who witnessed the incident smaller sums and had them sign similar documents.

He then revised the maintenance logs to show that the valve had been correctly inspected two weeks before the failure. It had not. The valve failure was predictable. Gould's inspection in 1882 had flagged the third processing line's valve maintenance schedule as borderline adequate. Oake had noted it and done nothing.

Thomas Birch died because Silas Oake did not want to spend the money on a valve replacement.

This is what Gould found in the three-day gap in the accident log. Not the gap itself — the shadow of the gap. A reference in an earlier entry to a note that should have followed it and did not.

---

## The Canal Path

The canal path runs alongside the sulphur storage facility. At night it is unlit and the path edge is unmarked at the point where the path bends closest to the water. Gould walked it alone. He had been walking it for four evenings. On Thursday, at approximately nine pm, he was not alone for the first part of the walk.

Oake did not plan this carefully. He planned it as carefully as a frightened man who has three days can plan anything. He went home first, so his wife would see him leave the house after supper as was his habit. He took the service road behind the sulphur storage rather than the main gate. He carried nothing. He needed nothing. The path did the rest.

---

## The Compensated Worker

James Whitlow was one of the two workers who witnessed the September incident and signed Oake's document. He has not honoured his silence as completely as Oake hoped. He has been seen near the works twice in the past month. He has not yet spoken to anyone official, but he has spoken in a public house about injustice in general terms that two people remember.

He had not yet completed signing his document — he received his payment but the agreement was still being prepared by Oake's solicitor. He therefore has no legal obligation to silence and knows it. He has a grievance that is entirely genuine and a presence near the works that looks, from the outside, like threat.

He is the red herring and also, eventually, the resolution. A LATE phase witness token carries his testimony — not about Gould's death, which he did not witness, but about September 1886. This testimony, when it surfaces, saturates `motive_dim` completely and explains why Gould had to die before he filed his report rather than after.

---

## Design notes

The mechanism is similar in structure to Case 06 (Tidal Interval): the killer uses a physical environment rather than direct force. The canal path at night, unlit, edge unmarked. Oake needed only to be there at the right moment. The difference from Case 06 is that Oake was present — he walked with Gould for the first part, which the canal path tokens establish through shoe prints and a button found at the bend. This makes the `mechanism_dim` slightly more direct than in Case 06.

The maintenance log revision is the case's most precise physical evidence. Not a forgery — a revision. The ink and handwriting differences are minor enough to pass casual inspection but not a twenty-year inspector's eye. This detail should be a MID phase OBJECT token: specific, technical, low temperature (certain). It is the moment Gould knew, which is the moment the case's clock started.

Whitlow is the most complex red herring in the set. His grievance is real, his behaviour is suspicious (presence near the works, public house comments), and he has partial legal exposure (the unsigned document). But his role is the opposite of suppression — he wants the truth known, which is why he kills no one and why his testimony eventually destroys Oake. Players who pursue Whitlow are pursuing a thread that leads to the same truth via a different route. The question is whether they arrive in time to understand the path they walked.

The city name is deliberately withheld. The case is set in the generalised industrial north — a geography defined by canals, chemical works, and the smell of processing — rather than a named location. This is a design choice that allows the atmosphere tokens to do more work than they would in a pinned geography.
