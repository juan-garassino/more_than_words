# Case 13 — The Observatory Clock
## Paris Observatory · Belle Époque · October · 1900

| | |
|---|---|
| **Case ID** | `observatory_clock` |
| **Difficulty** | hard |
| **Vocab** | 72 tokens |
| **Killer** | `suspect:professor_serin` |
| **Mechanism** | `event:ladder_failure` |
| **Motive** | `motive:survey_espionage` |
| **Red herring** | The observatory director — blocked Cord's promotion twice, institutional reputation depends on academic standing, documented antagonism |
| **False plateau** | Director path partially fills `killer_dim` and `motive_dim`; `mechanism_dim` breaks it — the director had no reason to target the specific ladder Cord used; acid damage requires close access and chemistry knowledge |
| **Bridge token** | The observation log anomalies — a TIME token cluster showing entries made during periods when cloud cover was recorded in the meteorological register; dates match German military survey schedules |

---

Mathieu Cord is twenty-seven years old and has been a junior astronomer at the Paris Observatory for three years. He is not a social success — he is too precise for conversation and too methodical for the informal currency of institutional life. He does not care about this. He has found, in the transit instrument's dome, a place where precision is the only thing that matters, and he has been happy there.

He submits his Congress paper on the twelfth of October. It demonstrates, using eighteen months of carefully replicated observations, that the published theorem on stellar aberration by Professor Édouard Serin contains a fundamental error in its treatment of atmospheric refraction at high zenith angles. The error is not minor. It invalidates a decade of cited work.

He is found at the base of the transit dome's ladder on the evening of the fifteenth.

---

## The Director

Fournier is fifty-six, the observatory's director for eleven years, a man whose professional life has been spent managing the gap between the institution's ambitions and its resources. He has blocked Cord's promotion twice — once on grounds of seniority and once on grounds of what he described to colleagues as "insufficient institutional perspective," by which he meant that Cord does not attend the right dinners or know when to be quiet.

He has reasons to want the Congress paper suppressed. The paper will embarrass the observatory. Serin is the observatory's most cited living astronomer. The correction will appear at an international gathering and will be reported. Fournier's name, as director, will appear in the coverage.

He is not the killer. He is the institutional weather that made the crime possible — the environment in which a junior astronomer's concerns could be dismissed, in which a problem with the dome ladder could go unreported for three visits, in which Serin's access to the instrument dome at any hour was unquestioned. The director's tokens establish the institution. They do not close the case.

---

## Professor Serin

Édouard Serin is fifty-eight and has been a senior astronomer at the observatory for twenty-two years. His theorem on stellar aberration was published in 1891. It was presented at three international conferences. It is cited in the reference tables of two standard astronomical almanacs. He has spent the intervening nine years building his reputation on its foundation.

There is a second thing Serin has been doing for the past four years. The transit instrument logs record it, if one knows how to read them.

Serin makes observations at times when observation is impossible. Not frequently — eleven entries in thirty months, scattered across the record. Each entry records a specific astronomical target at a specific time, with a precision that would require an exceptionally clear sky. The meteorological register, kept independently in the observatory's eastern tower, records cloud cover at those same times. Not partial cloud. Total cover. The dome would have been sealed.

The entries are not observations. They are signals.

---

## The Logs

Cord noticed the first discrepancy by accident, cross-referencing a weather correction to his own data. He noticed the second because the first had made him look. By the sixth he had constructed a table. By the tenth he had identified the pattern.

The dates of the impossible observations cluster. They repeat at intervals that do not correspond to any astronomical cycle Cord can identify. He spent two evenings in the meteorological archive and one evening with a German railway schedule that a colleague had brought back from a conference. The intervals match a survey reporting cycle. He wrote this down in a private notebook, not the official log. He did not tell anyone. He was not sure what he was looking at.

He had told Serin about the stellar aberration paper on the tenth. He had told no one about the log anomalies. He had left both his private notebook and his Congress paper draft in the dome on the evening of the fifteenth, intending to collect them after supper.

---

## The Ladder

The transit dome's ladder reaches the upper observation platform from the instrument floor. It is eleven meters of iron rungs set into a fixed frame. The rungs are inspected annually. Three of them — the seventh, eighth, and ninth from the top, where the weight distribution is most critical during descent — had been treated with dilute acid at the fixing points sometime in the previous three weeks. The treatment was careful. The corrosion it produced is indistinguishable from ordinary iron fatigue without close chemical analysis.

Serin has a private chemistry interest. He has a small laboratory in his residential quarters on the observatory grounds. He has been making his own photographic developing solutions for fifteen years.

The ladder was the only one giving access to the platform. Cord used it every clear evening. October has been exceptionally clear.

---

## The Time Cluster

The observation log anomalies are the case's pivot. They do not surface early — they require the meteorological register token, the German survey schedule token, and a witness token (a cleaning attendant who saw Cord's private notebook on the dome floor three days before his death) to assemble. When placed together, they form a TIME token cluster that saturates `motive_dim` and recontextualises the stellar aberration paper entirely.

The paper was not the original threat. The paper was a secondary problem that arrived at the worst possible moment. What Serin was protecting was not his academic reputation. That was already lost — the Congress paper would correct the theorem regardless of whether Cord was alive to present it. What he was protecting was the log. And the log was protecting something older than any of them: a contact, a reporting arrangement, a four-year transaction conducted in astronomical notation.

When `motive_dim` fills, the case closes on a man who killed a junior astronomer not to save his reputation but because the junior astronomer had, without knowing it, read his correspondence.

---

## Design notes

The stellar aberration paper is the red herring motive. It is real — Serin had real reasons to want it suppressed — and it fills `motive_dim` partially because the player can construct a coherent theory around it. The false plateau holds because this theory fits `killer_dim` and almost fits `motive_dim`, but cannot satisfy the `mechanism_dim` tokens, which insist on chemistry knowledge and repeated dome access.

The director path is shorter and more frustrating. It establishes institutional animosity without providing mechanism. Players who pursue it arrive quickly at a dead end and are redirected toward the instrument dome itself: the ladder, the access records, who worked in the dome at night.

The German survey connection is the case's most remote element. It is never named directly. The player assembles it from three tokens — impossible observations, meteorological records, an interval table — that individually point nowhere. Together they point to a pattern that Cord found by accident and Serin found by checking whether Cord had found it.

The observatory's clock room contains a master chronometer that synchronises the instrument logs. Time is what this case runs on: the time of entries, the time of cloud cover, the time of Cord's fall. Every key token has a timestamp. The TIME cluster that breaks the false plateau does so because clocks do not lie in the way people do. Serin could alter an entry. He could not alter the weather that was recorded independently.

The private notebook is never recovered. It was in the dome when Cord went up and was not found in the debris. Serin had time, before calling for help, to remove it. Its absence is noted in the police inventory. Its absence is itself a token.
