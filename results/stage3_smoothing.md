# Stage 3 smoothing

## Dev perplexity by method and order (default settings)

| method | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|
| add_k | 719.2 | 4,615.5 | 18,914.1 | 34,698.4 |
| jm | 241.7 | 145.9 | 141.6 | 151.8 |
| gt | 404.1 | 1,264.4 | 5,655.8 | 13,892.6 |
| katz | 219.0 | 137.1 | 125.1 | 126.0 |
| kn | 210.5 | 123.3 | 105.6 | 101.2 |
| mkn | 212.4 | 123.8 | 105.7 | 100.8 |

## Tuned on dev (order 5)

| method | perplexity | bits/char | known words only | setting |
|---|---:|---:|---:|---|
| add-1 (Laplace) | 88,754.1 | 3.575 | 87,233.6 | k=1 |
| add-k | 14,578.5 | 3.008 | 13,191.6 | k=1e-05 |
| interpolation | 127.5 | 1.521 | 130.5 | lam=0.45 |
| Good-Turing | 13,892.6 | 2.993 | 13,015.8 | none needed |
| Katz backoff | 126.0 | 1.517 | 125.3 | none needed |
| Kneser-Ney | 100.4 | 1.446 | 102.2 | D=0.8 |
| modified Kneser-Ney | 100.8 | 1.447 | 102.7 | estimated from data |

## Cross-check of our Kneser-Ney against NLTK (5,000 training sentences, D=0.75)

| order | tokens | ours | NLTK | mean difference | max difference |
|---:|---:|---:|---:|---:|---:|
| 2 | 789 | 295.63 | 295.63 | 0.0005% | 0.0052% |
| 3 | 789 | 282.75 | 282.75 | 0.0015% | 0.0580% |

## 10% vs 100% of the training data (order 5, dev perplexity)

| method | 10%: all tokens | 10%: known only | 100%: all tokens | 100%: known only |
|---|---:|---:|---:|---:|
| add-1 (Laplace) | 31,943.0 | 31,210.5 | 88,754.1 | 87,233.6 |
| add-k | 21,000.4 | 18,618.2 | 14,578.5 | 13,191.6 |
| interpolation | 273.1 | 331.6 | 127.5 | 130.5 |
| Good-Turing | 19,829.8 | 18,607.2 | 13,892.6 | 13,015.8 |
| Katz backoff | 232.7 | 268.1 | 126.0 | 125.3 |
| Kneser-Ney | 195.1 | 231.5 | 100.4 | 102.2 |
| modified Kneser-Ney | 194.4 | 230.6 | 100.8 | 102.7 |

## Tokenisations ranked by bits/char (modified Kneser-Ney, order 3)

| tokeniser | vocabulary | tokens/sentence | perplexity | bits/char |
|---|---:|---:|---:|---:|
| whitespace | 169,245 | 16.4 | 123.8 | 1.512 |
| lower_punct | 82,563 | 19.4 | 62.2 | 1.533 |
| punct | 99,223 | 19.4 | 64.9 | 1.549 |
| bpe16k | 15,937 | 18.7 | 92.8 | 1.625 |
| bpe4k | 3,998 | 22.1 | 48.6 | 1.646 |
| bpe3.5k | 3,499 | 22.7 | 44.8 | 1.653 |
| bpe3k | 2,999 | 23.4 | 40.7 | 1.663 |
| bpe2.5k | 2,499 | 24.4 | 36.2 | 1.679 |
| bpe2k | 1,999 | 25.9 | 30.9 | 1.706 |
| bpe1.5k | 1,500 | 28.8 | 24.7 | 1.770 |
| bpe1k | 1,001 | 41.0 | 14.4 | 2.100 |
| char | 853 | 76.2 | 6.9 | 2.814 |

## Unknown-word fix (order 5, all dev tokens)

| method | before | after |
|---|---:|---:|
| interpolation | 156.8 | 127.5 |
| Katz backoff | 126.0 | 126.0 |
| Kneser-Ney | 128.3 | 100.4 |
| modified Kneser-Ney | 128.0 | 100.8 |

## Cross-check of our modified Kneser-Ney against KenLM (full training split)

| order | KenLM: all | ours: all | KenLM: known only | ours: known only |
|---:|---:|---:|---:|---:|
| 3 | 156.72 | 156.82 | 124.03 | 123.87 |
| 5 | 128.25 | 128.04 | 101.09 | 100.44 |

## Bits per character by tokeniser and order (modified Kneser-Ney)

| tokeniser | order 2 | order 3 | order 4 | order 5 | order 6 | order 8 | order 10 | order 12 | best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lower_punct | 1.771 | 1.533 | 1.462 | 1.438 | 1.429 |  |  |  | 1.429 |
| whitespace | 1.681 | 1.512 | 1.462 | 1.447 | 1.443 |  |  |  | 1.443 |
| punct | 1.782 | 1.549 | 1.483 | 1.460 | 1.452 |  |  |  | 1.452 |
| bpe2k | 2.165 | 1.706 | 1.564 | 1.521 | 1.506 | 1.496 |  |  | 1.496 |
| bpe4k | 1.992 | 1.646 | 1.550 | 1.522 | 1.513 | 1.508 |  |  | 1.508 |
| bpe1k | 2.739 | 2.100 | 1.761 | 1.622 | 1.564 | 1.521 | 1.509 |  | 1.509 |
| bpe16k | 1.856 | 1.625 | 1.564 | 1.546 | 1.540 |  |  |  | 1.540 |
| char | 3.318 | 2.814 | 2.389 | 2.089 | 1.900 | 1.708 | 1.629 | 1.599 | 1.599 |
