# Case 11 — The Winter Station
## British Antarctic Research Depot · February · 1912

| | |
|---|---|
| **Case ID** | `winter_station` |
| **Difficulty** | medium |
| **Vocab** | 72 tokens |
| **Killer** | `suspect:dr_marsh` |
| **Mechanism** | `event:ice_crossing` |
| **Motive** | `motive:mining_concealment` |
| **Red herring** | Sayers, expedition cartographer — public argument over map credit, stands to benefit professionally from Hadley's death |
| **False plateau** | `killer_dim` and `motive_dim` fill on the Sayers path (argument, career benefit); `mechanism_dim` breaks it — Sayers had no plausible reason to be on the ice that night; the weak-section token connects to Marsh's equipment inspection route, not Sayers's |
| **Bridge token** | A letter begun but never sent, found in a frozen equipment cache — Hadley was writing to the London office about a South African mining connection |

---

Lieutenant James Hadley is thirty-one years old and has been a surveying officer for eight years. He is precise, thorough, and constitutionally incapable of leaving an inconsistency alone. These qualities have made him an exceptional surveyor. On the evening of February 17th they make him a dead man.

The Antarctic depot sits on a shelf of compacted snow eleven miles from the coast. The expedition's work is routine — magnetic readings, ice-core samples, the slow accumulation of data that will matter to someone in London in three years. The ice shelf extending southward from the depot's equipment caches is not routine. Three weeks ago Hadley noted, during a daytime traverse, that a section of it sounds different underfoot. Hollow. He flagged it in the survey log. He expected the expedition physician to confirm it when he led the cache inspection the following week.

Dr. Francis Marsh confirmed nothing. He recorded the section as stable.

---

## The Cartographer

Sayers is twenty-eight and has been drawing maps of places no one has seen before for the past four years. He is good at it. He is also aware, with the particular clarity of someone who has spent too many evenings in a small shared space with a larger personality, that Hadley's name appears above his own on the expedition's primary cartographic output — a survey of the western ice shelf that Sayers considers his own work in every sense that matters.

They argued about it on February 14th. The argument was witnessed by two other expedition members and was not quiet. Sayers said things he has been regretting since. Hadley said less but said it more precisely, which was worse.

Sayers is the obvious suspect. He had a grievance that the entire depot witnessed. He had professional reasons to want Hadley diminished — or absent. He is also, when the mechanism tokens surface, entirely impossible as the killer. He was in the chart room that evening. Three men saw him. He did not know which section of the ice was hollow, because Hadley had never discussed the survey anomaly with him. And he had no reason — none that any path through the field can construct — to be leading anyone across the southern ice shelf at eleven pm in February.

---

## Dr. Marsh

Francis Marsh is forty-four and has been an expedition physician for twelve years. He is calm under pressure and has an unusual knowledge of low-temperature physiology, which is why he was selected. He has also, for the past six years, been providing false health clearances to a South African mining company that extracts coal from a deep seam outside Johannesburg. The clearances certify workers fit for exposure levels that the company's own ventilation reports describe as unsafe. The company pays him well. The workers develop silicosis at twice the rate of comparable operations. Several of them have died.

Hadley found correspondence about this in Marsh's kit bag three days before he died. He should not have been in Marsh's kit bag. He was looking for a meteorological instrument Marsh had borrowed and not returned. He found a sheaf of letters from a firm called the Cresswell Mining Syndicate and a carbon copy of a medical declaration bearing a name he recognised from the expedition's casualty files.

He did not confront Marsh. He began a letter instead.

---

## The Ice Section

Marsh has known since his cache inspection which section of the southern approach sounds different underfoot. He knows because Hadley's survey log flagged it and he was asked to verify it. He recorded it as stable. He had no reason, in February, to care whether the section was stable.

He had reason on the night of February 17th.

Hadley had told him, that afternoon, that he needed to check on the cache situation himself — the equipment count from Marsh's inspection did not match the depot manifest. He said he would go after supper. Marsh offered to accompany him. He knew the route. He knew which lantern Hadley would carry and how far its light would reach. He knew the section of ice. He went out an hour ahead, without a lantern, and waited.

He did not push Hadley. He did not need to. He waited at the cache on the far side and called out that there was a problem with the stores. Hadley crossed toward the sound.

---

## The Letter

In the equipment cache on the southern approach, inside an oilskin pouch that Hadley had placed there himself on a previous traverse, investigators will eventually find six pages in Hadley's handwriting. They are addressed to the Secretary of the London Geographical Society, care of the expedition's administrative office. The letter is incomplete. It breaks off mid-sentence in a description of the Cresswell Mining Syndicate's operations in the Transvaal.

The letter establishes that Hadley knew about the correspondence. It establishes that he understood its significance. It establishes a connection — specific, named, dated — between Marsh and a mining operation whose workers have been dying.

The letter is a LATE phase OBJECT token. It surfaces cold — certain, specific, low temperature. When placed, it saturates `motive_dim` completely and reframes the entire preceding path. The argument with Sayers, which filled early dimensions convincingly, becomes a coincidence of timing. Marsh's inspection route, which connects directly to the weak-section token through affinity edges tagged "access, ice, knowledge," becomes the mechanism's spine.

---

## Design notes

The false plateau in this case works because Sayers's grievance is emotionally resonant and professionally plausible. Players who begin with the witnessed argument find `killer_dim` and `motive_dim` filling quickly. The plateau hits when the mechanism tokens refuse to support Sayers. He had no access to the hollow section. He had no route to the cache at night. He had no knowledge of where Hadley would go or when.

The redirect comes through the equipment inspection token — a MID phase OBJECT token establishing that Marsh led the cache inspection and that his recorded findings did not match Hadley's original survey notes. This discrepancy is the mechanism's first signal. It connects Marsh to the ice section and, once the letter surfaces, to the motive.

Marsh is effective as a killer because his role gives him legitimate reasons to be anywhere on the depot's operational perimeter. His equipment inspections, his medical rounds, his knowledge of the ice — all of these are professional cover that looks, from outside, like diligence. The player who begins with the letter arrives at the truth directly. The player who begins with Sayers must walk through the mechanism to get there.

The South African mining connection is deliberately remote. It is not a local grievance. It is a correspondence from another continent, a slow catastrophe that Marsh has been managing from a distance for six years. The motive tokens carry this distance — a company name, a carbon copy, a list of names in a medical file. The player assembles the scale of what Marsh was protecting from fragments, not from confession.
