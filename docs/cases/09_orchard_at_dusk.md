# Case 09 — The Orchard at Dusk
## Rural England · September · Harvest Season · 1903

| | |
|---|---|
| **Case ID** | `orchard_at_dusk` |
| **Difficulty** | easy |
| **Vocab** | 72 tokens |
| **Killer** | `suspect:thomas_crale` |
| **Mechanism** | `event:altered_path` |
| **Motive** | `motive:boundary_fraud` |
| **Red herring** | Amos Webb — witnessed threat, visible anger, no alibi, clearest surface motive |
| **False plateau** | Shallow; `motive_dim` cannot close on Webb because his grievance (the cheated survey) points toward Crale, not away; the buried motive (the 1884 transaction) recontextualises Crale's motive entirely |
| **Bridge token** | The 1884 transaction — oblique surface expression, two LATE phase motive tokens; when placed together they reveal that Crale killed to protect nineteen years of silence, not seven acres of orchard |

*Cross-case note: "Thornfield" appears here as the name of a village in Somerset. It remains only an in-world place name in this case.*

---

The boundary between Crale Farm and Webb Farm has been disputed since 1887. The dispute concerns seven acres of orchard — old apple trees, planted in the 1840s, whose productivity has increased significantly following drainage work completed in 1899. The orchard is worth arguing about in a way it was not before the drainage. Both families know this.

In August 1903 both families agreed to commission an independent land survey to resolve the dispute. The surveyor is a man named Frederick Oates, fifty-one, who travels from Bristol and arrives on a Monday. He is found in the long grass of the orchard path on Friday evening. The coroner records accidental death — a fall in the long grass, a blow to the head from a stone in the dry ground. It is consistent with misadventure.

It is not misadventure.

---

## The Survey

Oates measured incorrectly. Not through error — through instruction. Before he arrived at Thornfield he was contacted by Thomas Crale through a Bristol solicitor and offered a fee for a survey that would place the boundary favourably. Crale was careful. The conversation was indirect. The fee was paid through the solicitor as a "preliminary consultation charge." Oates understood perfectly.

He surveyed the boundary and placed it seven acres into Webb land. His report, filed on Wednesday, awards the orchard to Crale Farm.

Amos Webb — whose family holds the neighbouring farm — commissioned his own private measurement the following day. By Thursday evening he knows the survey is wrong. He confronts Oates at the village inn on Thursday evening. The confrontation is witnessed by four people. Webb threatens to report the fraud to the land registry. Oates, frightened and genuinely culpable, promises to file a correction. He will attend the registry office in Bristol on Monday.

He never reaches Monday.

---

## Crale

Thomas Crale is fifty-eight. He acquired the farm from his father in 1891 and has run it without distinction and without complaint for twelve years. He is not a violent man. He is a man with a secret that the survey was designed to protect.

The secret is this: the orchard was not simply land Crale wanted. It was land he was owed. The previous owner of what is now Webb Farm — a man named Alderton who died in 1898 — owed Crale a debt from a transaction in 1884 that neither of them ever recorded in writing. The transaction involved money that Crale should not have had, moved in a way that would have interested certain people if it had become known. Alderton knew this. The debt was their arrangement: silence on both sides, the land to settle it when the time came.

Alderton died before the transfer. His daughter had married into the Webb family, and Amos Webb inherited without knowing about the arrangement. Crale has been waiting for an opportunity. The boundary dispute and the drainage work created one.

The original transaction — the source of the money, the nature of what Crale should not have had in 1884 — is the buried motive. It surfaces through two LATE phase motive tokens that describe the 1884 period obliquely. When they surface the case recontextualises: Crale did not kill to keep seven acres. He killed to keep nineteen years of silence.

---

## The Gate

On Friday afternoon Crale moved a field gate on the orchard path. The gate's new position forced anyone walking the boundary at dusk — as Oates habitually did each evening to verify his measurements — to take a longer route through unmown long grass rather than the shorter mown path. The longer route passes within two meters of a drainage stone that sits above the grass level.

Crale was at supper with his farmhand at seven pm, which the farmhand confirms. He does not need an alibi for the gate. He moved it at four pm when the farmhand was in the upper field. The gate can be moved by one person in three minutes. There is no witness.

---

## Amos Webb

He is the most obvious suspect in the case and the player's companion through most of it. He had the clearest visible motive — the survey cheated him of seven acres. He was heard threatening the victim. He had opportunity. He is angry in a way that looks like guilt.

But Amos Webb is not the killer. He is something more interesting: he is closer to the buried truth than anyone else in the case without knowing it. His father-in-law Alderton's arrangement with Crale — the 1884 transaction — is something he has sensed the edges of without ever seeing it clearly. There is a token in his cluster, a LATE phase EMOTION token, whose surface expression is this: "Webb stood at the orchard fence for a long time after the argument. He was not looking at the trees. He was looking at the farmhouse."

He knew something was wrong before Oates arrived. He did not know what.

---

## Design notes

This is rated easy because the mechanism tokens are specific enough to exclude Webb quickly. He had no reason to move a gate. His method, if he had acted, would have been direct and witnessed. The surface of his anger points toward confrontation, not arrangement. The path through Webb is emotionally compelling but mechanically brief.

The ease is deceptive. The buried motive is the hardest in the ten cases to assemble precisely because it is the most historically remote — 1884 is nineteen years before the crime, and the tokens that carry it are oblique by design. The 1884 transaction is described only in terms of its structure (money, silence, land) not its content. What Crale was moving in 1884 is left open. This is a design choice: the engine requires only that the player place `motive:boundary_fraud` correctly, not that they understand every detail of what it conceals.

The orchard itself is a token cluster. The drainage work (1899), the changed productivity, the suddenly contentious boundary — these are EARLY phase tokens that establish why the survey happened at all. They are low weight individually but densely connected. A player who begins with the orchard rather than with Webb or Crale will take longer to reach the invariants but will understand the full geometry of the case when they arrive.

The farmhand's alibi for Crale covers supper, not the afternoon. This gap — the farmhand in the upper field, Crale alone at four pm — is a MID phase TIME token. It does not prove guilt. It proves possibility. Together with the gate token and the drainage stone token, it builds the `mechanism_dim` without requiring Crale to have been present at the moment of the fall.
