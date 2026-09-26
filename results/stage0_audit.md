# Stage 0 data audit

| Step | Sentences | Notes |
|---|---:|---|
| rows in | 4,408,322 | Ewe vocab 600,335 / English vocab 989,225 (vocab = number of different words) |
| after NFC | 4,408,322 | rows changed Ewe 0 / English 0; Ewe vocab 600,335 |
| after whitespace + Ð→Ɖ | 4,408,322 | Ewe rows changed 151,727; Ewe vocab 600,266 |
| deduped on Ewe (non-empty) | 995,588 | 77.4% of rows were duplicates |
| Ewe-filtered | 295,198 | rejection rate 70.3% of unique sentences |

## Splits (data/splits/<train>-<dev>-<test>/)

| Ratio | Train | Dev | Test |
|---|---:|---:|---:|
| 1/49/50 | 3,013 sent / 47,125 tok | 144,544 sent / 2,204,524 tok | 147,641 sent / 2,252,946 tok |
| 5/45/50 | 14,718 sent / 226,872 tok | 132,839 sent / 2,024,777 tok | 147,641 sent / 2,252,946 tok |
| 10/40/50 | 29,324 sent / 450,574 tok | 118,233 sent / 1,801,075 tok | 147,641 sent / 2,252,946 tok |
| 10/45/45 | 29,324 sent / 450,574 tok | 132,938 sent / 2,025,910 tok | 132,936 sent / 2,028,111 tok |
| 20/40/40 | 58,867 sent / 902,153 tok | 118,137 sent / 1,798,872 tok | 118,194 sent / 1,803,570 tok |
| 30/35/35 | 88,442 sent / 1,351,709 tok | 103,445 sent / 1,575,767 tok | 103,311 sent / 1,577,119 tok |
| 40/30/30 | 117,842 sent / 1,800,656 tok | 88,922 sent / 1,353,877 tok | 88,434 sent / 1,350,062 tok |
| 50/25/25 | 147,557 sent / 2,251,649 tok | 73,954 sent / 1,130,031 tok | 73,687 sent / 1,122,915 tok |
| 60/20/20 | 177,004 sent / 2,701,025 tok | 59,366 sent / 906,175 tok | 58,828 sent / 897,395 tok |
| 70/15/15 | 206,764 sent / 3,154,533 tok | 44,353 sent / 677,790 tok | 44,081 sent / 672,272 tok |
| 80/10/10 | 236,370 sent / 3,607,200 tok | 29,512 sent / 450,469 tok | 29,316 sent / 446,926 tok |
| 85/10/5 | 251,117 sent / 3,832,323 tok | 29,470 sent / 451,237 tok | 14,611 sent / 221,035 tok |
| 85/5/10 | 251,117 sent / 3,832,323 tok | 14,765 sent / 225,346 tok | 29,316 sent / 446,926 tok |
| 90/5/5 | 265,882 sent / 4,057,669 tok | 14,705 sent / 225,891 tok | 14,611 sent / 221,035 tok |
| 98/1/1 | 289,310 sent / 4,415,404 tok | 2,932 sent / 44,377 tok | 2,956 sent / 44,814 tok |
