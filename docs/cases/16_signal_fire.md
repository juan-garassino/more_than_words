# Case 16 — The Signal Fire
## Pacific Island · US Army Signals Post · November · 1943

| | |
|---|---|
| **Case ID** | `signal_fire` |
| **Difficulty** | hard |
| **Vocab** | 72 tokens |
| **Killer** | `suspect:captain_hayes` |
| **Mechanism** | `event:radio_modification` |
| **Motive** | `motive:deception_exposure` |
| **Red herring** | Rodrigo Santos — guerrilla commander whose fighters suffered losses traceable to Berger's supply decisions; he had a witnessed violent dispute with Berger and was present at the post two days before the death |
| **False plateau** | Santos path saturates `killer_dim` and `motive_dim` (grief, command responsibility, proximity); `mechanism_dim` breaks it — the radio modification requires signals expertise and knowledge of the frequency architecture specific to the AN/GRC-9 field unit; Santos had no access to this knowledge |
| **Bridge token** | OPERATION BLIND EYE authorisation cable — in Hayes's field kit, encoded; decrypted, it shows the false transmissions were authorised by Army intelligence six weeks before Berger's arrival on the island |

---

Thomas Berger arrives on the island on the fourteenth of October, seconded from OSS Manila. He is thirty-one years old, meticulous, experienced in signals analysis. His brief is to audit the post's transmission records for irregularities that might indicate Japanese interception. He is given full access to the logs. He is given four weeks.

He needs three.

On November 7th, Berger's body is found near the transmitter hut. Cause of death: burns consistent with a phosphorus ignition, originating from the radio casing. The AN/GRC-9 field unit's frequency selector mechanism, when rotated past 18.4 MHz — a frequency not used in standard operational protocols — triggers a small phosphorus charge embedded in the selector housing. The charge has been there for days. The modification is neat. Whoever made it knew the radio's internal architecture precisely.

Berger had been using that frequency to prepare a test transmission verifying his findings. He had told no one what frequency he intended to use. He had written it in his log.

---

## Captain Hayes

David Hayes is thirty-eight. He has commanded the signals post for seven months. Before that he ran signals operations in New Guinea; before that he trained at Fort Monmouth and knows the AN/GRC-9 series from its circuit diagrams. He is quiet, organised, and well-regarded by his men, who find his evenings solitary but not strange — he has always worked late.

Hayes does not deny that he modified the radio. He is not asked about it, because the investigation does not reach him. The preliminary report names equipment failure as the probable cause. Hayes reads the report the same afternoon it is filed and says nothing.

He knows exactly what he did. He has known since the morning he made the modification. He has not slept more than four hours in any night since.

---

## The Operation

OPERATION BLIND EYE is an authorised signals deception programme coordinated between Army intelligence and Filipino guerrilla command. Its purpose is to feed false grid coordinates to Japanese communications interception — coordinates that, when acted on, draw patrol activity away from guerrilla supply lines in the interior. Hayes has been running it since March. He has a written authorisation, encoded, in his field kit. He has never shown it to anyone at the post because the operation's security requires that it not be acknowledged.

Thomas Berger, working through the transmission logs with the rigour his training required, identified the false coordinates within two weeks. He did not know they were false in the way Hayes knew them to be false. He knew only that they were wrong — that they did not correspond to Allied positions and that they were being transmitted on a secondary frequency at irregular intervals that matched Japanese interception windows. He concluded, correctly but fatally, that Hayes was transmitting to the enemy.

Berger was preparing a report for OSS Manila. Hayes discovered this on November 5th when Berger's field log — left open on a table in the signals hut — showed the draft opening line: *Irregular transmissions identified, command attention required, evidence of deliberate enemy communication.*

The report, if transmitted to Manila, would reach a Philippine Army liaison officer named Aquino who Hayes knew — through a separate intelligence channel — was a Japanese asset. Aquino would transmit the exposure within hours. The Japanese would understand that BLIND EYE was compromised. They would reverse their patrol movements. The guerrilla supply routes, stripped of deceptive cover, would be exposed. Hayes's contact at guerrilla command had estimated two hundred fighters in those lines.

Hayes tried to reach his intelligence handler for thirty-six hours. The channel was down — a relay station in Leyte had been destroyed in a Japanese air strike on November 3rd. He had no way to warn Manila through secure channels. He had no way to explain to Berger without breaking the operation, and breaking the operation would accomplish the same thing Berger's report would accomplish. He had two days before Berger's scheduled transmission window.

---

## Commander Santos

Rodrigo Santos commands the primary guerrilla unit operating from the island's interior. He has been fighting since 1941 and has lost forty-three men, several of them in actions where the supply shortfalls Berger's office approved contributed to the outcome. Santos knows this. Berger knew it too. Their argument, on October 29th, was witnessed by four men and lasted twenty minutes. Santos's position was that Berger's allocation decisions valued strategic abstraction over the lives of specific fighters. Berger's position was that Santos was confusing command responsibility with personal culpability.

Neither position was entirely wrong.

Santos was at the post on November 5th — he had come to arrange a resupply meeting. He was there again on November 6th. On the 7th, when Berger's body was found, Santos was in the interior. He has an alibi that is both genuine and unprovable. He has a motive that is both real and insufficient.

The `mechanism_dim` is what exonerates him. The phosphorus charge required three things: knowledge of the AN/GRC-9 frequency architecture, access to the transmitter hut's tool cabinet, and a minimum of four hours of unobserved work on the unit. Santos has none of these. He had never been inside the transmitter hut. The frequency architecture of Army field radios was not part of his operational knowledge. He is a guerrilla commander, not a signals engineer.

The player who follows Santos follows a thread that is morally coherent but physically impossible. The mechanism closes it.

---

## The Authorisation Cable

Hayes's field kit contains a standard signals kit: spare components, a codebook for his current operational period, maintenance manuals for the equipment in his charge. Among the maintenance manuals, filed between pages 34 and 35 of the AN/GRC-9 operator guide, is a folded document. It is encoded in Army intelligence's standard cipher for the period.

Decoded — using the codebook in the same kit — it reads as an authorisation for a signals deception programme designated BLIND EYE, dated March 4th, 1943. It names Hayes as the operating officer. It specifies the false coordinates and the transmission frequencies. It is signed by a colonel in Army intelligence whose name appears on no document in Berger's investigation file.

The cable is a LATE phase OBJECT token. Its surface expression is: *an encoded document in a field maintenance manual.* It becomes legible only after the codebook token has been placed — MID phase, carried by a description of Hayes's kit contents that surfaces through the investigation of the transmitter hut. These two tokens, placed in sequence, form the case's pivot.

Before the cable surfaces, Hayes looks like a traitor. After it surfaces, Hayes looks like a man who killed to protect something real, using a method that was precise and premeditated and wrong.

The game names him correctly. The invariants confirm it. What the cable makes unavoidable is that Hayes was right about the threat and right about the timeline and right that Berger's report would have reached Aquino and right that the guerrillas would have died. He was not right about the method. The method is what the game is about.

---

## Design notes

This case is the second in the series to close with moral ambiguity — the first being Case 10 (The Attended Hour). Like that case, the field closes correctly: three invariants identified, proof gate passed. Like that case, the close raises a question the engine cannot answer.

Hayes is the case's most complex killer in the set. He did not act from fear for himself. He acted from a calculation about other lives, made with incomplete information and no good options, in the forty-eight hours before he believed two hundred people would die. The calculation was plausible. The conclusion was not one the game can endorse. It names him. The player decides what that means.

The `motive_dim` token cluster should carry both the surface reading (treason cover-up, self-protection) and the buried reading (BLIND EYE exposure, guerrilla massacre) as distinct token layers. The surface reading saturates by turn eight. The buried reading requires the authorisation cable and the Aquino intelligence token — a LATE phase WITNESS token whose surface expression is *a name in a separate report, mentioned once.* Only when Aquino's identity as a Japanese asset is placed alongside the cable does `motive_dim` complete.

The Santos path is the case's most emotionally plausible false trail. His grief is genuine. His anger at Berger was witnessed. His presence at the post was recent. Players who follow him are not making an error of reasoning — they are following evidence that legitimately points toward him until the mechanism breaks it. This is the correct design. The red herring should sustain eight turns of serious consideration. Santos sustains ten.

Positioning: Case 16 should follow cases that have established moral clarity. It should not be played back-to-back with Case 10. The series needs cases between them where guilt is unambiguous — where the field closes on someone who had no justification. Hayes needs that context to register correctly.
