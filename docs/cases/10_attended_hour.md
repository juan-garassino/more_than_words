# Case 10 — The Attended Hour
## Private Hospital · Cardiac Ward · Night Shift · November · Present Day

| | |
|---|---|
| **Case ID** | `attended_hour` |
| **Difficulty** | hard |
| **Vocab** | 72 tokens |
| **Killer** | `suspect:callum_dray` |
| **Mechanism** | `event:dosage_interaction` |
| **Motive** | `motive:negligence_testimony` |
| **Red herring** | None. This case has no innocent suspects. Both Callum and Mara are guilty of something. The question is which guilty act is the killing one. |
| **False plateau** | `killer_dim` and `mechanism_dim` saturate by turn nine; `motive_dim` stalls at 0.48 because Callum's personal fear is only half the buried truth; it cannot close until Mara's three-am decision surfaces |
| **Bridge token** | The three-am chart note — Mara's routine round, her encounter with the medication combination, the training record that confirms she recognised it; placed together, these tokens reveal a second decision made in full knowledge |

*Design note: This is the final case by design. It is the most morally complex and the least clean. The field closes correctly — three invariants identified, proof passed — but it closes on a question rather than an answer. This is intentional.*

---

The patient's name is Richard Ayres. He is fifty-eight, post-surgery, recovery expected but complicated by an irregular medication response that has required the night team's attention twice this week. His surgeon considers him stable. His cardiologist is mildly concerned. His solicitor has been contacted twice this week about scheduling a deposition.

Ayres is the primary witness in a medical negligence case involving this ward. The case has been in preparation for two years. His deposition is scheduled for Thursday. He is found unresponsive on Wednesday at three forty-five am. Resuscitation attempts are made for twenty-two minutes. He dies at four-oh-seven.

The death is recorded as cardiac complications consistent with his existing condition and his medication protocol. Nothing in the record is wrong. Everything in the record is accurate. The record was designed this way.

---

## Callum

Callum Dray is thirty-four and has been a charge nurse on this ward for six years. He is competent, calm under pressure, well-regarded by colleagues, and has never in his professional life acted outside protocol in a way that could be documented.

He adjusted Richard Ayres's medication dosage at two am on Wednesday. The adjustment was within recordable parameters — it falls inside the acceptable range for the medication given Ayres's weight and recent bloodwork. What the record does not show is that Ayres is also receiving a secondary treatment for his irregular response, and the interaction between the two medications at the dosage Callum selected produces a sustained increase in cardiac stress. This interaction is documented in the pharmacological literature. It is not commonly known. Callum knows it because he looked it up.

He has been a peripheral figure in the negligence case for eighteen months. The case concerns an event from three years ago in which a patient — not Ayres — died following a procedure where several staff members signed documentation they had not personally verified. Callum was one of those staff members. He signed because his supervisor asked him to and because the patient was already critical and because it was two am and he had been on shift for fourteen hours. He has not slept well since.

Ayres's deposition would not have named Callum as the primary actor. It would have named him as a witness who signed documentation without verification. Enough to end his nursing registration. Enough to end his career. Enough, possibly, for a criminal charge.

---

## Dr. Lund

Dr. Mara Lund is forty-seven, the attending physician on the night shift. She checks Ayres's chart at three am as part of her rounds. She notes the medication combination. She understands its significance immediately — she has encountered this interaction once before, in a training context, and it stayed with her.

She records nothing. She continues her rounds. She is back at the nursing station at three-fifteen. She does not raise an alert. At three forty-five the monitor sounds and she is the first physician to respond.

She had her own reason to want Ayres's deposition not heard. The negligence case concerns an event she was present for. She was not a signatory to the documentation. She was, however, the physician who advised that the documentation process could be abbreviated given the patient's critical state. This advice was informal, verbal, unrecorded. Ayres was present when she gave it. He remembers it. She knows he remembers it.

She is not the killer. She is the reason the killer was never investigated.

---

## The Two Decisions

This case turns on the difference between two acts that look, from outside, identical.

Callum adjusted the dosage at two am knowing the interaction and its probable effect. This is the killing act.

Mara read the chart at three am knowing the interaction and its probable effect, and continued her rounds. This is not the killing act. But it is the act that allowed the killing to complete, and it was made in full knowledge by a person with the training to understand exactly what would happen.

The game names Callum correctly. But `motive_dim` does not saturate until the player has assembled Mara's three-am decision as a second layer of the motive. Callum killed Ayres. Mara let it happen. The motive token `motive:negligence_testimony` carries both of them — the deposition would have named both of them, in different ways, for the same original event. The motive is shared. The act was Callum's alone.

---

## The Shape of the Close

Most paths converge by turn nine on: Callum adjusted the dosage, the interaction killed Ayres, the deposition was the reason. This is the surface truth. It is correct. The invariants confirm it.

`Motive_dim` stalls at 0.48. The player has the surface motive — Callum's fear of the deposition — but not the buried structure. The buried structure is that Callum's vulnerability was not isolated. Mara shared it. Her silence at three am was a decision made by someone who understood the stakes, which means her silence was not passive. It was chosen.

The LATE phase tokens that carry this: Mara's training record (her encounter with the medication interaction in a formal context), the three-am chart note (timestamped, routine-looking, the one place where her knowledge is documented), and a witness token that places her at Ayres's room at three-oh-five — two minutes longer than a standard check. Two minutes in which she stood at the chart and read it again.

When these three tokens are placed, `motive_dim` moves to completion. The field closes.

The final three tokens are Callum Dray, the medication interaction at two am, and the negligence testimony. The case closes correctly.

What the game cannot answer — and does not try to — is where Callum's guilt ends and Mara's begins. The engine identifies the killer. The player decides what that means.

---

## Design notes

This case was designed last and should be played last. The other nine cases build toward a world where guilt is locatable and singular. Case 10 presents a world where guilt is distributed and institutional. The engine still works — three invariants, proof gate passes — but the moral resolution is deliberately incomplete.

No innocent suspects is a deliberate design inversion. In every other case, the red herring character is sympathetic and unambiguously innocent. Here, both primary figures are guilty of something. This forces the player to engage with degree rather than presence of guilt, which is harder and more uncomfortable.

The mechanism is the most medically precise in the set. The dosage interaction is real — two specified medications at specified parameters producing a documented cardiac stress response. The token cluster for `event:dosage_interaction` should be the most technically specific cluster in any case: medication name, dosage, interaction parameter, cardiac response pathway. Low temperature (certain), high weight on `mechanism_dim`.

Callum's original act — signing documentation he had not verified — was not monstrous. It was tired and pressured and institutional. The motive tokens should carry this without excusing it. He is not a monster who killed a man. He is a person who made a small, exhausted compromise three years ago that grew into something that required a larger one. The game shows the field. The player decides what to feel.
