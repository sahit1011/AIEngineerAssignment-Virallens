Held-out chunks: 72 (train: 279)

### Labels: topic

| metric | baseline | lsa | domain |
|---|---|---|---|
| silhouette | 0.191 ± 0.000 | 0.256 ± 0.001 | 0.307 ± 0.007 |
| ari | 0.307 ± 0.012 | 0.769 ± 0.327 | 0.462 ± 0.138 |
| nmi | 0.523 ± 0.039 | 0.857 ± 0.202 | 0.615 ± 0.076 |
| knn_purity | 0.956 ± 0.000 | 0.982 ± 0.001 | 0.962 ± 0.007 |
| similarity_gap | 0.144 ± 0.000 | 0.206 ± 0.001 | 0.226 ± 0.002 |

### Labels: doc_type

| metric | baseline | lsa | domain |
|---|---|---|---|
| silhouette | 0.189 ± 0.000 | 0.247 ± 0.001 | 0.308 ± 0.011 |
| ari | 0.292 ± 0.011 | 0.708 ± 0.284 | 0.468 ± 0.174 |
| nmi | 0.522 ± 0.039 | 0.784 ± 0.146 | 0.612 ± 0.101 |
| knn_purity | 0.939 ± 0.000 | 0.967 ± 0.000 | 0.951 ± 0.003 |
| similarity_gap | 0.149 ± 0.000 | 0.211 ± 0.001 | 0.230 ± 0.001 |

### Labels: doc

| metric | baseline | lsa | domain |
|---|---|---|---|
| silhouette | 0.114 ± 0.000 | 0.208 ± 0.001 | 0.196 ± 0.010 |
| ari | 0.346 ± 0.030 | 0.507 ± 0.011 | 0.433 ± 0.007 |
| nmi | 0.556 ± 0.033 | 0.677 ± 0.007 | 0.614 ± 0.008 |
| knn_purity | 0.728 ± 0.000 | 0.820 ± 0.001 | 0.781 ± 0.016 |
| similarity_gap | 0.134 ± 0.000 | 0.224 ± 0.001 | 0.201 ± 0.001 |

### Retrieval MRR: 6 eval questions answered by a held-out passage, ranked against the 72 held-out passages

| baseline | lsa | domain |
|---|---|---|
| 0.774 ± 0.000 | 0.853 ± 0.001 | 0.695 ± 0.035 |

Questions: macquarie-holding, kogan-quiz, overdisclosure, concentration-policy, fb-positions-compared, macquarie-in-facebook

### Domain model training loss (mean per epoch)

- seed 0: 1.900, 1.212, 0.951, 0.863
- seed 1: 1.871, 1.204, 0.955, 0.866
- seed 2: 1.883, 1.185, 0.908, 0.895

### KMeans (k = 3) cluster composition by topic (seed 0)

- baseline: cluster 0: disclosure_omissions 33, specialist_execution 9; cluster 1: disclosure_omissions 16; cluster 2: disclosure_omissions 1, fund_concentration 12, specialist_execution 1
- lsa: cluster 0: disclosure_omissions 50; cluster 1: fund_concentration 12; cluster 2: specialist_execution 10
- domain: cluster 0: disclosure_omissions 31, specialist_execution 1; cluster 1: fund_concentration 12, specialist_execution 9; cluster 2: disclosure_omissions 19

### Nearest held-out neighbours (seed 0)

**macquarie p. 11**: * * * Pure omissions are not actionable under Rule 10b-5(b). The judgment of the Court of …
- baseline (1/3 same filing): macquarie p. 11 (0.678), fb-petitioners p. 23 (0.573), fb-petitioners p. 36 (0.518)
- lsa (1/3 same filing): macquarie p. 11 (0.545), fb-petitioners p. 23 (0.337), last-atlantis p. 10 (0.314)
- domain (1/3 same filing): macquarie p. 11 (0.796), fb-united-states p. 10 (0.661), last-atlantis p. 11 (0.653)

**fb-petitioners p. 10**: Petitioners are Facebook, Inc., now known as Meta Platforms, Inc., which operates the soci…
- baseline (2/3 same filing): fb-petitioners p. 48 (0.799), fb-petitioners p. 35 (0.784), fb-united-states p. 11 (0.779)
- lsa (2/3 same filing): fb-petitioners p. 35 (0.847), fb-petitioners p. 48 (0.721), fb-united-states p. 11 (0.717)
- domain (2/3 same filing): fb-petitioners p. 35 (0.849), fb-united-states p. 11 (0.844), fb-petitioners p. 10 (0.838)

**fb-united-states p. 10**: 229.105(a). That discussion “must be organized logically with relevant headings” and with …
- baseline (1/3 same filing): fb-petitioners p. 11 (0.58), fb-united-states p. 34 (0.575), fb-petitioners pp. 23-24 (0.572)
- lsa (2/3 same filing): fb-united-states p. 36 (0.351), fb-petitioners pp. 23-24 (0.256), fb-united-states pp. 36-37 (0.243)
- domain (2/3 same filing): fb-united-states p. 34 (0.663), fb-petitioners p. 23 (0.629), fb-united-states p. 35 (0.621)

**last-atlantis p. 10**: With no Seventh Circuit case law directly on point , I will3 follow the Second Circuit’s h…
- baseline (3/3 same filing): last-atlantis p. 10 (0.801), last-atlantis p. 11 (0.733), last-atlantis p. 10 (0.661)
- lsa (3/3 same filing): last-atlantis p. 10 (0.747), last-atlantis p. 11 (0.717), last-atlantis p. 11 (0.705)
- domain (3/3 same filing): last-atlantis p. 10 (0.884), last-atlantis p. 11 (0.863), last-atlantis pp. 12-13 (0.838)

**sec-v-chiueh p. 10**: 28. A mutual fund’s portfolio is managed by a Commission-registered investment adviser. A …
- baseline (3/3 same filing): sec-v-chiueh pp. 10-11 (0.76), sec-v-chiueh pp. 11-12 (0.571), sec-v-chiueh p. 22 (0.47)
- lsa (3/3 same filing): sec-v-chiueh pp. 10-11 (0.641), sec-v-chiueh pp. 34-35 (0.462), sec-v-chiueh pp. 11-12 (0.44)
- domain (3/3 same filing): sec-v-chiueh pp. 10-11 (0.8), sec-v-chiueh pp. 34-35 (0.641), sec-v-chiueh pp. 11-12 (0.616)
