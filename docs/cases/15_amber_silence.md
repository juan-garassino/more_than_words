# Case 15 — The Amber Silence
## Occupied Normandy · Village of Saint-Clair · 1943

| | |
|---|---|
| **Case ID** | `amber_silence` |
| **Difficulty** | hard |
| **Vocab** | 72 tokens |
| **Killer** | `suspect:henri_dumont` |
| **Mechanism** | `event:bell_tower_door` |
| **Motive** | `motive:milice_protection` |
| **Red herring** | SOE agent codenamed "Oiseau" — met with Élise the previous day; when the killing is discovered it presents as a German security action against a Resistance contact |
| **False plateau** | SOE path fills `killer_dim` and `mechanism_dim` partially; `motive_dim` breaks it — British agents in 1943 Normandy did not kill French Resistance contacts through staged accidents; the method points toward a local who knew the bell tower's access pattern; ACCESS tokens redirect to Dumont |
| **Bridge token** | Élise's lesson plan book — she transcribed the German officer's letter in small handwriting between French grammar exercises; the book is in a drawer of her classroom desk |

---

Élise Renard — no relation to the Renard Voss of Case 01 — is thirty-four years old and has been teaching in Saint-Clair's village school for seven years. She is known to three people in the village as something more than a teacher. She is known to none of them as a woman who kept records.

The bell tower of Saint-Clair's church has an external access door on its eastern face. The bell-ringer's route uses a different door, from inside the church. The eastern door is used to access the bell-rope housing for maintenance, which happens twice a year, and otherwise stands locked. Élise had a copy of the key. She had had it for two years.

On the morning of November 4th she is found in the lane below the tower's eastern face. The Gendarmerie record her death as an accidental fall. The German garrison commander notes it in his administrative log as a possible security incident. The village buries her on November 6th, in rain. Henri Dumont delivers a brief eulogy. He taught her confirmation class twenty years ago.

---

## Oiseau

The SOE agent who had met with Élise the previous afternoon is a British officer, thirty years old, inserted six weeks earlier to establish courier routes in the Calvados region. He is using the name Marcel. He is not using the name Oiseau in his own presence — it is his network designation, known to London and to the three contacts he has established so far. Élise was the most recent.

When the German garrison administrator begins an inquiry into her death, the meeting with a stranger the previous day surfaces quickly. A neighbour saw them in the churchyard for twenty minutes. The description of the stranger is sufficient to confirm, to anyone who knows what to look for, that a British agent was active in the village. The Gestapo officer who arrives from Caen on November 7th is entirely certain, within an hour of his arrival, that Élise was killed by the Germans — either by the garrison to suppress a network, or by a German counterintelligence operation that has already been documented in other villages.

This interpretation is wrong. It is wrong in a way that benefits Dumont, who has been counting on it since the morning of November 4th.

The SOE path produces a coherent account for the first eight turns. The meeting, the contact, the pattern of staged accidents used against Resistance figures elsewhere in occupied France. The mechanism tokens refuse it. A British agent in 1943 Normandy does not remove a door hinge pin and wait. British security actions against suspected informers were direct when they occurred at all. The specificity of this mechanism — a single hinge pin, on a specific door, at a specific time — requires local knowledge of a domestic kind. The kind that takes years.

---

## Henri Dumont

Henri Dumont is sixty-two. He has been mayor of Saint-Clair for fourteen years. Before the Occupation this was a civic function with modest duties. Since 1940 it has been a position of careful navigation, and Dumont has navigated. He has protected the village from the worst requisitions. He has managed the German garrison's demands without precipitating reprisals. He is not, in the village's collective judgment, a collaborator in any shameful sense. He is a man doing a difficult job.

His son Pascal left for Lyon in March 1942. He returned in June 1942. He said nothing about what he had done in Lyon. Henri did not ask. But a German officer — a Gestapo administrative officer named Brandt, stationed in Caen — sent Dumont a letter in September 1942, in the official language of administration, referencing Pascal's "continued useful service" and noting that his enrollment documentation was being held pending a formal decision about military affiliation. The letter said, without saying, that Pascal's file would remain in administrative limbo as long as his father remained cooperative.

Henri has been cooperating since September 1942.

Élise found Brandt's letter in October 1943. She found it not by searching Dumont's papers but because Dumont had asked her to retrieve a document from the mairie's filing cabinet while he was occupied with a garrison visit, and the letter was in the wrong folder. She read it in thirty seconds and replaced it. She did not tell Dumont she had read it.

She began transcribing it from memory that evening, in the only safe place she could think of.

---

## The Bell Tower Door

The eastern access door opens inward from outside. It is heavy oak, original to the tower, and its iron hinges are a century old. Élise had used this door thirty times in two years. She knew its weight and its swing.

Dumont had the parish maintenance keys. He had had them for fourteen years. He removed the hinge pin — the lower pin, the one that takes the door's weight — on the evening of November 3rd, after the curfew cleared the streets. The pin was seated loosely but held. The door would open and close normally. It would behave normally for the first several uses. Under full door weight during a descending exit — when a person stepped back against the door to pull it shut from the outside — the lower hinge would give.

Élise used the tower every Thursday after dark. She had been using it every Thursday for eighteen months. Dumont knew this because his bedroom window faces the church, and because he had known what she was doing with the key since the first winter she used it, and because until October 1943 he had considered her work necessary and her discretion reliable.

He went to bed before curfew on November 3rd. He was seen at his front door at nine pm by two neighbours. He did not leave the house again, visibly, until morning.

---

## The Lesson Plan Book

Élise's classroom is in the village school, two streets from the church. After her death it is locked. The school does not reopen until a replacement teacher arrives, which takes three weeks. No one searches the classroom because no one knows there is anything in it to find.

Her lesson plan book is a standard issue educational register, green-covered, her name printed on the inside front cover in her own hand. The week of November 3rd shows a grammar exercise — French conditional tenses — with examples written in two columns. Between the columns, in pencil, in handwriting that is not French grammar notation but is superficially similar enough to look like correction marks, is a transcription of Brandt's letter. Forty-seven words. The date, the reference to Pascal's service, the phrase about enrollment documentation. Brandt's name. The return address in Caen.

The book token is LATE phase. It surfaces through a chain: a witness who saw a light in the school on an evening when Élise was not teaching (she was there making the transcription), a colleague's memory of Élise commenting that she had found something that worried her, and the school administrator's inventory of uncollected classroom materials. When placed, it saturates `motive_dim` completely and breaks the SOE interpretation finally and without ambiguity.

Dumont killed her not because she suspected him of informing — that suspicion, she had expressed to the priest, could have been managed, denied, weathered. He killed her because she had read the letter. Because she knew about Pascal. Because Pascal's file in a German administrative office was worth more to him than the schoolteacher's life.

The shame of it is not the cooperation. The shame is what he decided the cooperation was worth.

---

## Design notes

This case is rated hard because the false plateau is unusually stable. The Oiseau path fills two dimensions convincingly and has documentary support — the German garrison log, the Gestapo officer's arrival, the pattern of occupied-France security actions against Resistance contacts. The mechanism tokens break it, but slowly: the specificity of the hinge-pin removal requires local knowledge that the ACCESS tokens distribute gradually across the MID phase.

The motive is the case's deepest layer. `Motive:milice_protection` surfaces its buried content only when the lesson plan book is found. Before that, the player can construct a theory around Dumont as an informer protecting himself. This is partially true — he was cooperating — but it does not scale to murder. The Pascal connection scales. A father protecting a son from a war crimes record, in 1943, with liberation on the horizon and the future of documented collaboration becoming suddenly real — this scales.

Dumont is the most morally complicated killer in the set after Case 10. His cooperation was not enthusiastic. He is not a fascist. He is a man who made a decision about his son and has been paying for it every month since. The motive tokens carry this weight — the mairie administrative records, the cooperative posture, the management of requisitions — before they reveal the letter. When they do reveal it, the player is asked to hold two things simultaneously: what Dumont protected, and what he destroyed to protect it.

The SOE agent "Oiseau" is the only red herring in the set who was operating legitimately and dangerously in the same environment as the crime. This is what makes the false plateau hold. He is not an innocent bystander mistaken for a killer. He is an active combatant whose presence and purpose naturally produce the appearance of a security action. Players who pursue him are not wrong about the world they're in. They are wrong about who acted in it, on this specific night, with a hinge pin and a door.
