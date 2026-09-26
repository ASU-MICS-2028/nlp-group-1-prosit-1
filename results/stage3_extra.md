# Stage 3 extras

## A. Charging word-level models for spelling unknown words

| tokeniser | order | bits/char, flat unknown cost | bits/char, unknown words spelled | unknown tokens |
|---|---:|---:|---:|---:|
| whitespace | 6 | 1.443 | 1.627 | 5,392 |
| lower_punct | 6 | 1.429 | 1.516 | 2,345 |
| punct | 6 | 1.452 | 1.551 | 2,836 |
| bpe2k | 8 | 1.496 | 1.496 | 0 |
| bpe4k | 8 | 1.508 | 1.508 | 0 |

## B. Small tokenisers at higher orders (bits/char)

| tokeniser | order 8 | order 10 | order 12 | order 14 | order 16 |
|---|---:|---:|---:|---:|---:|
| bpe1k | 1.521 | 1.509 | 1.507 | 1.507 |  |
| bpe2k | 1.496 | 1.495 | 1.495 |  |  |
| char | 1.708 | 1.629 | 1.599 | 1.590 | 1.590 |
