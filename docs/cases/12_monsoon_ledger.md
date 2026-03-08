# Case 12 — The Monsoon Ledger
## Calcutta · Partition of Bengal Agitation · 1905

| | |
|---|---|
| **Case ID** | `monsoon_ledger` |
| **Difficulty** | hard |
| **Vocab** | 72 tokens |
| **Killer** | `suspect:gerald_peel` |
| **Mechanism** | `event:quinine_substitution` |
| **Motive** | `motive:uprising_concealment` |
| **Red herring** | Basu, Hindu nationalist activist — documented violent threats against the newspaper, publicly named Ghosh as an enemy; threats are on file with the police |
| **False plateau** | Basu's path fills `killer_dim` and surface `motive_dim`; `mechanism_dim` breaks it — quinine substitution requires access to her medicine cabinet; the ACCESS token redirects to Peel, who visited her office under official cover |
| **Bridge token** | The 1857 census extract — filed in the newspaper's research materials; Ghosh's assistant holds instructions about its disposition in the event of her death |

---

Amrita Ghosh is forty-three years old and has been editing the Nayak for eleven years. The paper has survived suppression orders, printing press seizures, and two cycles of editorial staff departing for safer employment. It has survived because Ghosh is meticulous about what she publishes and why, and because she keeps records of everything. Her office has three locked drawers. She keeps the keys on her person.

She dies in the first week of the monsoon, when the city is loud and the drains are overwhelmed and no one is thinking about anything except the rain. Her death is recorded as fever complications — she has had malaria for four years and has been managing it with quinine sulfate prescribed by her physician. The prescription is current. The medication bottle is found on her desk. The medication in the bottle is not quinine sulfate.

---

## The Activist

Basu is thirty-two, a printer's son who became a pamphleteer and then, following the Partition announcement, something more organised and considerably more dangerous. He has been threatening the Nayak in print for six months, in terms that the city's police commissioner has described in correspondence as "actionable." The threats concern the newspaper's secularist editorial line — Ghosh has published pieces critical of communal violence on all sides, which Basu considers a betrayal of the cause. He named her specifically in a pamphlet distributed three weeks before her death.

The police have his file. The commissioner has recommended action twice. The threats are documented, specific, and public.

Basu did not have access to her medicine cabinet. He has never been inside her office. He does not know her daily routine, her physician's name, or the name of the compound that would replace her medication without immediate detection. He is a printer's son who makes pamphlets. He is also, by the time the mechanism tokens surface, entirely impossible.

He is not innocent of intention. He is innocent of this particular act.

---

## Gerald Peel

Assistant Commissioner Peel is forty-one and has been in the Bengal Civil Service for sixteen years. He is methodical, well-regarded in Calcutta's administrative circles, and has been his father's keeper in the particular way that second sons become keepers — quietly, completely, without ever discussing it. His father, Colonel Archibald Peel, served as a junior officer during the 1857 Uprising. He was promoted twice in the years following. He died in 1891 with a distinguished service record and a pension his widow still draws.

Gerald has been drawing the pension for her since 1899. He manages his father's papers.

Amrita Ghosh was an editor. She was also a historian's correspondent, and the historian — a retired professor at Presidency College — had told her about the census records of 1857. Not the published records. The administrative records. The ones that documented the disposition of a particular village in a particular district in late 1857, and the officers responsible for what was documented there. Ghosh spent four months tracing the originals. She found them in the Bengal Secretariat archive, misfiled under a survey notation that was not quite right.

She had not yet published. She was checking her sources.

---

## The Two Visits

Peel visited the Nayak twice in the month before Ghosh's death. Both visits were official — the first to deliver a notice about a content restriction under press regulations, the second a follow-up inspection required by the same regulation. Both were documented. Both were unremarkable.

On the second visit he was in the office alone for eleven minutes while Ghosh was called to the pressroom to resolve a compositor's dispute. Her desk was unlocked. The medication bottle was on the desk. He had prepared a substitute compound the previous weekend, in the dispensary of the civil hospital where his position gave him access to pharmaceutical stores. The substitute is a febrifuge — it addresses fever without addressing the Plasmodium parasite. In a patient with established malaria, it manages symptoms while the underlying infection advances.

He replaced the bottle and left. He signed the inspection register on the way out. His handwriting is neat.

---

## The 1857 Extract

Ghosh's assistant is a woman named Priya Chaudhuri, twenty-six, who has been with the paper for two years. She does not know what is in the locked drawer. She knows that Ghosh gave her a sealed envelope eighteen months ago and told her to take it to the professor at Presidency College if anything happened to her. Priya has kept the envelope in her satchel since then. She takes it to the professor four days after Ghosh dies.

The envelope contains a copy of the 1857 census extract. It also contains a letter in Ghosh's handwriting explaining what she had found and where the original is filed. The professor, reading this, understands what it means. He also understands what it means that Ghosh is dead.

This token is LATE phase. It surfaces slowly, through the assistant's testimony, the professor's account, and the archive index entry that confirms Ghosh had requested the original three months before she died. When placed, it saturates `motive_dim` completely. The newspaper corruption story — which Peel could have managed through administrative pressure — was the cover. The 1857 records were the reason.

---

## Design notes

The difficulty comes from the mechanism's invisibility. Quinine substitution leaves no physical trace that a 1905 physician would recognise. The cause of death is genuine fever. The medication bottle is on the desk. The prescription is current. Nothing is wrong with the record. This makes `mechanism_dim` the hardest to fill, because its key token — the ACCESS token placing Peel alone in the office — is MID phase and arrives long after the Basu path has already filled two dimensions convincingly.

Basu is the case's most effective red herring because his guilt is documented and official. The police file on his threats is a real object. Players who begin with it are following genuine evidence to a genuine threat. The redirect requires the mechanism tokens to insist: this substitution required pharmaceutical knowledge, prescription access, and a period alone with the victim's personal desk. Basu satisfies none of these.

The buried motive is layered. The surface motive — Ghosh was publishing corruption — is true and sufficient to explain why Peel would want her silenced through administrative means. It is not sufficient to explain murder. The 1857 records are the reason murder became necessary. They are also older than anything else in the case: forty-eight years old, misfiled, forgotten by everyone except the archivist who filed them and the historian who eventually thought to ask.

The motive token `motive:uprising_concealment` carries both layers. Its surface expression names a family and a record. Its affinity edges connect to the archive, to the census extract, to the professor's institution, and to a payment record — Colonel Peel's 1857–1858 promotion documentation — that the player never sees in full but whose shadow is present in the family papers tokens.

Peel is not a man who wanted to be a murderer. He is a man who spent thirty years managing a secret and then found, in the first week of monsoon, that his management had run out.
