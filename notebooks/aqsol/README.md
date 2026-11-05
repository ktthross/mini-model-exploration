# What the models learned about solubility

Twelve notebooks, in pipeline order, on AqSolDB's aqueous solubility (logS). They were written to find how many bits a readable model can carry about solubility. Read as one argument, they answer a different question: the models learn a rule about logP and size that has been known since 1980, plus a lookup of neighbouring compounds, and nothing else the representation lets them learn. This file is that argument, with the notebook and number behind each step. The notebooks themselves are unchanged; the first ten score models on four solubility tiers, the last two on the value itself (the tiers were a misreading of the floor, see step 5).

## The line

**1. A measurement carries about 1.4 bits, and the noise is sources disagreeing.** Of 9,946 compounds, 1,580 have two or more independent measurements. After collapsing copies between databases and flagging gross errors (6% of votes in well-measured compounds, typically 2 logS off), a single measurement carries 1.42 bits about its compound's consensus value; the replicate noise has a robust SD of 0.59 logS and heavy tails. That is a compilation's agreement with itself, not a property of nature. (01, 11)

**2. Two descriptors carry the information: logP and size.** In every descriptor set the forest's two largest losses under shuffling are MolLogP and the group holding molecular weight; nothing else comes close (04). The best single question is *MolLogP ≤ 2.76*, the same threshold on all ten splits (09). Every equation symbolic regression finds, offered four inputs or eight, is logP and one size measure, as a bend rather than a line (08). The additive model's logP curve is a sigmoid with 2.7 times the importance of the next term, and size saturates by 250 Da (07). This is the general solubility equation's logP term (Yalkowsky & Valvani 1980) with a size correction. Its missing term is the crystal, through melting point, which no 2D descriptor sees.

**3. The rest is a lookup.** Twenty nearest neighbours on the descriptors give 0.86 of the forest's 1.01 bits with no function fitted at all (06). An additive model gives 0.94, and the pairwise interactions it finds are properties of the fold, not of the data: ten fits chose 110 distinct pairs and none in every fit (07). Structural fingerprints, atom environments and fixed message passing never beat 152 descriptors, and neighbours on fingerprints share less than neighbours on descriptors (04, 06). The forest's confidence in a prediction is how much its trees agree, which is how dense the training neighbourhood is (12).

**4. Where the rule fails, nothing helps.** Learning to Split collects the 10% of compounds whose measured values contradict what logP and size imply. They are as structurally familiar as a random test fold (nearest training neighbour at Tanimoto 0.55 against 0.59) and no single descriptor marks them (05). On those compounds every model is at or below zero information: MLP, forest, nearest neighbours, additive model, equations, on tiers and on the value (05, 06, 07, 08, 11, 12). No fingerprint sees a difference the descriptors miss: by every representation, the compounds structurally like them sit in another tier, and the measured value disagrees with all of them (06). Of the 912 compounds every seed found hard, 809 have a single measurement, so the data cannot say whether the value or the chemistry is wrong.

**5. Against the floor, scored on the value itself.** The repeat measurement reaches 97% of the floor. The forest reaches 90% on random splits, 83% on cluster splits, and about zero on the learned splits; with per-compound bars from its tree spread, 96% and 90%, and still zero (11, 12). The random-split number is interpolation among neighbours. The small drop on cluster splits is the logP-and-size rule transferring, as a rule does. The collapse on the learned splits is total because the rule and the lookup are all there is. The floor itself is a landscape: compounds the trees agree on have replicate noise near 0.47 logS, the ones they disagree on 0.70 to 0.77 (12).

**Verdict.** On this data, with 2D descriptors and fingerprints, the models learn logP, size, and who their neighbours are. They have no way to say why two similar molecules differ in solubility, and when the measured value says they do, the models are confidently wrong by every architecture tried. What is missing is not in the representation: the solid form (melting point), the ionization state, and for most of the hard compounds, a second measurement.

## The notebooks

| | notebook | role on the line | key result |
|---|---|---|---|
| 01 | `01_prep_and_analysis` | identity, votes, flags, the floor | 1.5 bits per measurement (KDE on consensus); 6% gross errors |
| 02 | `02_tiers` | superseded framing, kept as a pipeline step and as the cost of discretization | four tiers cost 0.5 bits before any model ran |
| 03 | `03_splits` | random and cluster splits, grouped by skeleton | cluster splits: no test compound has an identical fingerprint in training |
| 04 | `04_features` | which representation, which descriptors | 152 descriptors, 1.01 bits; MolLogP and MolWt dominate every set |
| 05 | `05_learning_to_split` | the hardest 10%, and what they have in common | familiar chemistry, no univariate signature, 912-compound core |
| 06 | `06_knn` | neighbours as the no-function model; representation vs architecture | 0.86 bits; the core is the odd one out in every representation |
| 07 | `07_additive` | the readable model that keeps the bits; the shape functions | 0.94 bits; logP sigmoid; interactions unstable |
| 08 | `08_equations` | the shortest model | 0.6 bits in 22 nodes; logP and one size term, a bend |
| 09 | `09_trees_and_rules` | the provably best small tree | 0.56 bits; one question, MolLogP ≤ 2.76 |
| 10 | `10_front` | information vs description length, tier axis | readable up to 0.95 of the forest's 1.01 |
| 11 | `11_continuous` | the same models scored on logS against the floor | floor 1.42; forest 1.28 / 1.18 / < 0 |
| 12 | `12_confidence` | per-compound bars and a 0 to 1 confidence score | 96% of floor; noise is heteroscedastic; `data/aqsoldb_predictions.csv` |

Shared scoring lives in `scripts/aqsoldb_harness.py` (tier and continuous). Each notebook reads the previous ones' hand-offs from `data/`; run them in order on the project `.venv` (see `docs/`).

## What this does not settle, and where the next chapter starts

The hard compounds are where a label, the solid, or the ionization state differs from what the structure implies, and the structure alone cannot tell which. Three ways in, each needing data this project did not use: the source of each value and its conditions (temperature, pH, solid form), a replicate measurement for the single-vote core, or an experimental melting point so the general solubility equation can be tested in full. The ruler built here, a floor, calibrated densities and a confidence score, is what makes any of those checkable.
