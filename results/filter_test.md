# Filter tests

## A. Religious-text skew

35.0% of training sentences contain a Bible/church keyword or start with a verse number (12.8% start with a verse number). This is a lower bound.

## B. GlotLID (a real language identifier) vs our letter filter

- Of the 295,198 sentences our filter KEPT, GlotLID calls 92.4% Ewe.
- Of the 700,390 it REJECTED, GlotLID calls 10.7% Ewe.
- Of our dev set, GlotLID calls 92.1% Ewe.

Top languages among kept: ewe_Latn 272,893, gaa_Latn 2,769, ajg_Latn 2,480, nzi_Latn 1,370, akp_Latn 1,257, ada_Latn 1,159, dyi_Latn 800, men_Latn 799

Top languages among rejected: ewe_Latn 74,923, pcm_Latn 53,357, bew_Latn 52,702, eng_Latn 13,909, kiu_Latn 13,555, dag_Latn 12,929, nrm_Latn 12,525, nap_Latn 12,028

Kept by us, not Ewe per GlotLID (sample):

- JW January 2019 Na'am Mɔɔlegɔ Yetɔɣum — gur_Latn (0.96)
- George Young vale nyevile zo fane dɔɔnwo dule adenle bɔle edwɛkpa nolo — nzi_Latn (1.0)
- Maybe ʌ lɔkɔ lɔm — men_Latn (0.51)
- "Ɛseta Dule Ɔ Nwo Maanle Gyihova Nee Ye Menli Ne": (Mit. — nzi_Latn (1.0)
- ge>ge: lulɔ — tod_Latn (0.77)
- Nu Enɛ: Nwīekpo kpa ama aara "le yereue loo lenu" nyɔnɛbee a lu yereue a le bu Baibol. — ogo_Latn (1.0)
- $1 vaseɖe $2 — und_Lina (0.37)
- Kɛ́ amrɔ nɛɛ ojeee gbɛgbalɔ lɛ, mɛɛ tsakemɔi obaanyɛ ofee koni obatsɔ gbɛgbalɔ? — gaa_Latn (1.0)
- Shi ni owo lɛ, ye nɔ." - Jajelɔ 5:4 — gaa_Latn (1.0)
- Gbɔmɛi kome - — gaa_Latn (0.99)
- Namɔ Ji Nuu ni Woloŋmaa Tɔ Ŋmɔ Ehɛ Lɛ? - Ezekiel 9:2 _ Nikasemɔ — gaa_Latn (1.0)
- 59 Yː Ni yɔɔ dzaanɔ̄ ... — gaa_Latn (0.93)

Rejected by us, Ewe per GlotLID (sample):

- Late Lammas 4 kpl (0.39)
- Zigbee Enabled Tablets _ eBay (0.49)
- 9 eye dzi beads power (1.0)
- Wo, wo, wo, un moment, Mme la Présidente. (0.5)
- Archive - You make me feel? (0.25)
- Egbe Vado - Alone Soul uploaded by Egbe Vado - Listen (0.76)
- Tso - Nafusi (1.0)
- Vanessa Paradis Joe Le Taxi Meme (0.64)
- Send private message to agbadza (0.57)
- 'Nu wowwie Wunnie, id nu be wong time.' (0.32)
- My Two Cents: Home Again, Home Again (0.37)
- Eye Am That Eye Am: The Hipnotic Eye (0.85)

## C. Filter on vs off (modified Kneser-Ney, order 3, scored on the filtered dev set)

| training set | sentences | vocabulary | dev perplexity | known words only | bits/char |
|---|---:|---:|---:|---:|---:|
| letter filter (ours) | 265,882 | 169,245 | 123.8 | 126.6 | 1.512 |
| no filter | 896,230 | 558,960 | 155.2 | 161.8 | 1.583 |
| GlotLID filter | 313,315 | 186,700 | 124.4 | 129.1 | 1.513 |
