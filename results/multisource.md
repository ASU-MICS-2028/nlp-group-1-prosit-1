# Three more Ewe sources

Modified Kneser-Ney, order 5, whitespace tokens. Each source is split with the main split's hash rule; sentences already in the main corpus are removed.

| source | raw lines | new unique sentences | train / dev / test | religious share | passes the letter filter |
|---|---:|---:|---|---:|---:|
| bible_csv | 28,614 | 18,156 | 16,300 / 931 / 925 | 22.0% | 94.9% |
| dictionary | 540 | 367 | 327 / 18 / 22 | 15.5% | 99.2% |
| speech | 19,151 | 19,150 | 17,213 / 982 / 955 | 0.0% | 100.0% |
| web (main corpus) | | | 265,882 / 14,705 / … | 12.8% | 100% (filtered) |

## Bits per character on each source's dev set (lower is better)

| trained on | web | bible_csv | dictionary | speech |
|---|---:|---:|---:|---:|
| web only (main path) (265,882) | 1.447 | 1.304 | 1.519 | 2.119 |
| web + three sources (299,722) | 1.422 | 1.242 | 1.525 | 1.699 |

## Perplexity on known words, and unknown-word rate

| trained on | web | bible_csv | dictionary | speech |
|---|---:|---:|---:|---:|
| web only (main path) | 102.7 (2.4%) | 66.7 (3.1%) | 138.4 (4.0%) | 1315.4 (11.0%) |
| web + three sources | 94.2 (2.3%) | 54.1 (1.6%) | 139.2 (3.0%) | 249.6 (3.1%) |
