# Case 06 — The Tidal Interval
## Island Research Station · October Storm Season · Present Day

| | |
|---|---|
| **Case ID** | `tidal_interval` |
| **Difficulty** | medium |
| **Vocab** | 72 tokens |
| **Killer** | `suspect:piers_dunne` |
| **Mechanism** | `event:tidal_window` |
| **Motive** | `motive:fishing_economy` |
| **Red herring** | Dr. Marcus Webb — documented grievance, present on island, no alibi |
| **False plateau** | Shallow; mechanism tokens do not support Webb (no path access, no tidal expertise beyond common knowledge, no physical motive); redirect to logistics layer is fast |
| **Bridge token** | The path marker — wooden post with yellow reflective band, found eleven meters from its original socket hole, with fresh soil on its base |

*Cross-case note: Dr. Sarah Okafor is no relation to James Okafor, the conservatory student of Case 05. The shared surname is a coincidence of the world.*

---

The Tern Island Marine Research Station has eleven permanent staff, a diesel generator that fails in heavy weather, and a satellite uplink that works reliably except when it matters. It sits on the eastern edge of an island that exists primarily because of fishing. The island has four hundred and twelve residents. Three hundred of them have a family member on a commercial fishing vessel.

Dr. Sarah Okafor — no relation to the conservatory student — is the station's lead marine biologist. She has been here for six years. She has spent four of those years conducting a survey of the marine protected area surrounding the island. Her survey is complete. Her conclusions are clear. The protected area must be extended by forty percent. The extension will close the most productive fishing grounds within forty miles.

She knows what this means for the island. She has spent two years trying to find an alternative interpretation of her data. There is no alternative interpretation. She submits the final report to the station director on Friday afternoon. He approves it for external submission on Monday morning.

She goes for her evening walk on Sunday and does not return.

---

## The Eastern Path

The path to the eastern rocks runs along an unfenced cliff edge for three hundred meters. In summer it is walked daily. In October storm season the rocks are wet, the path edge is unclear in poor light, and the tidal interval between passable and impassable at the far end is forty minutes. Everyone on the island knows this. It is the kind of knowledge that is so fundamental it is never stated.

Sarah walked this path every evening at seven pm. She had done so for six years. She found it helped her think.

---

## Piers

Piers Dunne is forty-one and has been the station's logistics coordinator for eight years. He manages supplies, schedules, equipment maintenance, and — because the island is small and roles accumulate — the station's interaction with the local fishing community. He is good at this. He is trusted by both sides. His family runs the largest fishing operation on the island — three vessels, twelve employees, a processing facility on the harbour that his father built and that Piers has been paying the bank for since his father's stroke in 2019.

The operation is over-leveraged. The loans were taken on the assumption that the marine protected area would remain as currently defined. A forty percent extension triggers a statutory review within ninety days. The review will close the fishing grounds. The operation fails within six months. The processing facility goes. The twelve employees go. His father's life's work goes.

Piers had access to the station's movement schedules — he manages them. He had the tidal data — he maintains the station's equipment. He knew Sarah's routine because he had coordinated resupply schedules around it for years. He knew the window.

He did not push her. He was not there. He altered the path marker at the fork two hours before her walk — moving it twelve degrees, enough to send someone unfamiliar with the exact route in poor light toward the shorter unfenced section rather than the longer safe one. Sarah was not unfamiliar with the route. But it was a dark evening, the storm was building, and she was thinking about Monday morning.

---

## The Inspector

Dr. Marcus Webb arrives on the island on Thursday to conduct a routine review of the station's data collection methodology. He has a professional disagreement with Sarah about her statistical approach that he has aired in two journal letters in the past year. He believes her conclusions are sound but her methodology is selectively presented in a way that obscures certain variables. He said so to her face on Friday evening, at length, in the station's common room, in front of four witnesses.

He is on the island on Sunday. He has no alibi more specific than his room. He had a documented grievance with the victim. He is the most obvious suspect in the case and is entirely innocent.

The path through Webb is short — four or five turns — because the mechanism tokens don't support him. He had no access to the path marker. He had no knowledge of the tidal interval that was not general knowledge shared by everyone on the island. He had no motive that required her death rather than her professional embarrassment. Players who pursue Webb reach a dead end quickly and are redirected toward the logistics layer: the schedule data, the tidal records, the path marker itself.

---

## The Path Marker

The physical evidence in this case concentrates in one object: a wooden post with a yellow reflective band, found eleven meters from its original socket hole on Monday morning when the search party retraced Sarah's likely route. The post has fresh soil on its base. The original socket hole has compressed earth at its rim that suggests recent removal rather than weathering failure.

This token is LATE phase and high weight on `mechanism_dim`. When it surfaces it connects directly to the logistics coordinator token through the affinity edge tagged "access, maintenance, physical." The field shifts hard.

---

## Design notes

The mechanism in this case is the most passive of the ten: Piers never touched Sarah. He moved a post. The storm and the cliff did the rest. This passivity is what makes the case medium difficulty rather than easy — the mechanism tokens describe an indirect chain (moved marker → wrong path → unfenced section → storm) rather than a direct act.

The tidal interval is the case's structural clock. The forty-minute window between passable and impassable is a constraint that everyone on the island internalises but never articulates. The TIME token that captures this is what connects Piers's knowledge (logistics, schedules, tidal data) to the mechanism (the specific post, moved at the specific time). A killer who did not have Piers's role could not have known which forty-minute window mattered.

The motive carries the island's economics as a system, not just Piers's personal situation. His father's stroke, the processing facility's debt, the twelve employees — these are not background. They are the weight that makes the motive token high-temperature and ambiguous. The player places `motive:fishing_economy` and the surface expression names what is at stake for an entire community, not just one man.

Dr. Webb's name recurs in this case and Case 09 (Orchard at Dusk, where Amos Webb is the red herring farmer). These are different people in different centuries sharing a common English surname. The repetition is not narrative — it is world-building texture.
