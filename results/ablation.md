# Ablation: leave one preparation step out

Modified Kneser-Ney, order 5, whitespace tokens. Dev perplexity (lower is better).

| variant | train sentences | vocabulary | dev perplexity | known words only | bits/char | note |
|---|---:|---:|---:|---:|---:|---|
| baseline (all steps) | 265,882 | 169,245 | 100.8 | 102.7 | 1.447 | the Stage 0 pipeline |
| without NFC | 265,882 | 169,245 | 100.8 | 102.7 | 1.447 | NFC changed 0 rows, so this is identical |
| without whitespace cleanup | 266,817 | 169,245 | 100.8 | 102.6 | 1.447 | extra/leading spaces kept; ' s' and 's' no longer merge in dedupe or the split key (different dev set) |
| without the Ð→Ɖ fix | 249,693 | 164,209 | 103.2 | 105.3 | 1.456 | Ð-spelled words stay separate vocabulary entries (different dev set) |
| without dedupe | 2,303,964 | 169,245 | 142.6 | 136.0 | 1.556 | training keeps all copies (a hub sentence 1,000+ times); dev is deduped so the exam is the same |
| random split (leakage) | 280,493 | 174,358 | 93.1 | 94.5 | 1.424 | 352 of 14,705 dev sentences (2.4%) have a near-copy in train (different dev set) |

Also measured elsewhere: the Ewe filter (results/filter_test.md: no filter = 25% worse) and smoothing (results/dev_and_smoothing_checks.json: plain counting = infinite perplexity on 81% of dev sentences).
