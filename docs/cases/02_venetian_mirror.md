# Case 02 — The Venetian Mirror
## Private Palazzo · Venetian Lagoon · Carnival Week · 1931

| | |
|---|---|
| **Case ID** | `venetian_mirror` |
| **Difficulty** | hard |
| **Vocab** | 72 tokens |
| **Killer** | `suspect:countess_morvaine` |
| **Mechanism** | `event:balcony_fall` |
| **Motive** | `motive:forgery_operation` |
| **Red herring** | Count Aurelio Ferri — documented grievance, witnessed argument, no alibi |
| **False plateau** | `killer_dim` and `mechanism_dim` saturate around turn 8; `motive_dim` stalls at ~0.55 because the surface motive (protect the fraud) is visible but the buried motive (protect the political network) does not surface until LATE phase |
| **Bridge token** | The 1919 insurance valuation — connects Faretti's discovery to the original sale of the originals, which leads to the network |

---

The Palazzo Morvaine sits on a minor canal in the Dorsoduro sestiere, three bridges from the Accademia. It has been in the Morvaine family for two hundred years and contains, according to its insurance valuation, one of the finest private collections of Flemish Renaissance painting outside of Belgium. The valuation was conducted in 1928. It was conducted by a man who owed the Countess a favour. Every painting in the collection is a forgery.

Not amateur forgeries. Extraordinary ones. The work of a single forger — a Flemish restorer named Henrik Claes who died in 1924 and whose technique was so precise that three of his copies hang in the Uffizi catalogue without remark. The Countess acquired the originals in 1919, sold them privately to a consortium of buyers whose names are distributed across four countries, and replaced them with Claes's copies before the first insurance assessor set foot on the premises. She could not sell them through legitimate channels — doing so would require authentication that would expose the copies still hanging on her walls.

The money from the originals funds an operation the Countess has been committed to since 1916 — a political network operating across six European capitals whose purpose the player never learns in full but whose existence is suggested by the embassy tokens that surface in the late game.

---

## The Restorer

Marco Faretti is forty-three, Florentine, and very good at his work. He arrives on the Monday of carnival week to conduct a new insurance assessment — the Countess's insurer has requested an independent verification following a change in policy terms. Faretti does not know the collection is forged when he arrives. He knows by Tuesday afternoon.

He knows because of a single detail. One of the paintings — a small Bruegel-school winter scene — has a craquelure pattern that is internally inconsistent. The aging of the paint has been artificially accelerated in the lower left quadrant. Faretti has seen this technique once before, in a work that later proved to be a Claes copy. He says nothing on Tuesday. He spends Wednesday morning at the Marciana library looking at insurance records from 1919. He sends a telegram on Wednesday afternoon. He does not say to whom.

On Thursday evening, during the masked procession along the canal, he falls from the second-floor balcony of the Palazzo Morvaine.

---

## The Countess

Elspeth Morvaine is sixty-one. She was born in Edinburgh, married a Venetian count at twenty-two, and has outlasted him by twenty-nine years. She speaks four languages without accent and has never in her adult life acted in haste. She acted in haste once, on a November evening in 1916, and the decision she made that night is the root cause of everything that follows in 1931. The player never sees that decision directly. They see its shadow in the motive tokens — a faded military commission, an unsigned letter in a language that is not Italian, a payment record from a bank that no longer exists.

She weakened the balcony railing herself, with a small chisel, on Tuesday night. She has been a restorer's widow for twenty-nine years and knows exactly how much structural material must be removed for a railing to hold under normal use and fail under sudden weight.

---

## The Carnival

This is the mechanism's genius. During the masked procession the canal-side of the palazzo is lined with two hundred people in costume. The noise is extraordinary. No one is looking at the balcony. Everyone on the balcony is masked. Faretti was not masked — he was watching the procession from above, which is why he went to the balcony at all, which the Countess knew because she invited him to watch from there, which is why the railing on that specific section was weakened and no other.

The fall looks like an accident. A man leaning too far over a railing on a cold evening during carnival. The railing was old. The palazzo is old. These things happen.

---

## The Rival

Count Aurelio Ferri is fifty-five and has been trying to acquire two specific paintings from the Morvaine collection for eleven years through legitimate purchase. He believes they are originals. He has been publicly and repeatedly refused — the Countess cannot sell because a legitimate sale would require authenticating the works, which would expose the copies. He argued with Faretti at a reception on Tuesday evening — a sharp exchange about attribution methodology that three people witnessed and two remembered clearly. He has no alibi for the hour of the fall because he was alone in his hotel room writing letters, which is exactly what a guilty man would say.

He is not guilty. He is a collector with bad luck and worse timing. But his presence in Venice during carnival week, his argument with the victim, and his documented desire for specific paintings in the collection make him the field's dominant red herring for the first half of any playthrough.

The player who pursues Ferri will not be wasting their turns — the path through him eventually leads to the insurance records of 1928, which lead to the original 1919 valuation, which surfaces the forger's name, which breaks the false plateau entirely.

---

## The False Plateau

The false plateau in this case is specific. Around turn eight, most paths converge on a coherent-seeming explanation: Faretti was killed by the Countess to protect the insurance fraud. This feels complete. The convergence meter reads high on `killer_dim` and `mechanism_dim` but `motive_dim` stalls at around 0.55.

It stalls because the player has the surface motive — protect the forgeries — but not the buried one. The buried motive is not insurance fraud. The insurance fraud is a funding mechanism. The buried motive is the network, the 1916 decision, the political operation that fifteen people across Europe depend on and that exposure of the forgeries would unravel. The motive tokens that carry this information are all LATE phase. They do not surface until the killer and mechanism dimensions are nearly saturated.

When they do surface the case recontextualises completely. The Countess did not kill a man to protect a fraud. She killed a man to protect something she believes in more than her own freedom. Whether the player judges this differently is not the game's concern. The game only asks them to find the truth. What they feel about it is their own.

---

## Design notes

The 1919 insurance valuation is both the bridge token and the case's structural spine. It connects: Faretti's discovery → the collection's history → the original sales → the consortium buyers → the political network. A player who finds the 1919 valuation in the Marciana library records has everything they need to begin assembling the buried motive.

The embassy tokens in the late game are deliberately vague. They establish the network's existence without naming its purpose. This is intentional — the Countess's motive is the protection of a belief, not a plan, and beliefs resist precise description. The LATE phase motive tokens should be high-temperature (ambiguous, open) rather than low-temperature (specific, certain).

Ferri's red herring works because it is structurally complete in two of three dimensions. The player who pursues him builds a real explanation that simply cannot account for the mechanism: Ferri had no access to the specific railing, no reason to weaken one balcony section and not another, no way to know Faretti would use that section on that evening. The mechanism tokens force the redirect.
