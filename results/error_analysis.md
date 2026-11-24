# Error analysis (rule-based tags)

Training fraction 1. Accuracy (%) is mean +/- std over seeds; delta is GCN minus BERT. Tags are heuristic keyword/structure rules, not manual annotation.

## Train restaurant -> test restaurant (in-domain), seeds [42, 43, 44]

| Category | n | BERT acc | GCN acc | delta |
|---|---|---|---|---|
| all | 1120 | 83.1 +/- 0.9 | 83.6 +/- 0.4 | +0.5 |
| contrast word | 160 | 84.6 +/- 2.5 | 87.1 +/- 2.5 | +2.5 |
| gold = negative | 196 | 78.1 +/- 1.0 | 79.4 +/- 4.2 | +1.4 |
| gold = neutral | 196 | 44.0 +/- 1.5 | 42.7 +/- 2.5 | -1.4 |
| gold = positive | 728 | 94.9 +/- 0.9 | 95.7 +/- 0.2 | +0.8 |
| long sentence (>25 words) | 153 | 84.5 +/- 1.6 | 85.8 +/- 1.4 | +1.3 |
| mixed polarity in sentence | 228 | 66.5 +/- 3.3 | 68.0 +/- 0.4 | +1.5 |
| multi-aspect sentence | 835 | 83.0 +/- 1.0 | 83.4 +/- 0.5 | +0.4 |
| negation | 206 | 75.9 +/- 1.5 | 77.7 +/- 2.6 | +1.8 |

GCN right on all seeds and BERT wrong on all seeds: **2** examples; the reverse: **2**.

**GCN right, BERT wrong** (first 2):

| Sentence | Aspect | Gold | Wrong prediction | Tags |
|---|---|---|---|---|
| The portions of the food that came out were mediocre. | portions of the food | neutral | negative | - |
| It took 100 years for Parisi to get around to making pizza (at least I don't think they ever made it before this year)...but it was worth the wait. | pizza | positive | neutral | contrast word, long sentence (>25 words), negation |

**BERT right, GCN wrong** (first 2):

| Sentence | Aspect | Gold | Wrong prediction | Tags |
|---|---|---|---|---|
| You can eat gourmet food at a fast food price. | price | positive | neutral | multi-aspect sentence |
| " The menu includes pub fare--burgers, steaks and shepherds pie--and even a portabella lasagna for those black sheep known as "vegetarians. | menu | neutral | positive | multi-aspect sentence |

## Train restaurant -> test laptop (cross-domain), seeds [42, 43, 44]

| Category | n | BERT acc | GCN acc | delta |
|---|---|---|---|---|
| all | 638 | 74.6 +/- 0.8 | 75.0 +/- 1.3 | +0.4 |
| contrast word | 76 | 59.2 +/- 3.5 | 63.2 +/- 1.3 | +3.9 |
| gold = negative | 128 | 88.0 +/- 3.9 | 89.1 +/- 2.1 | +1.0 |
| gold = neutral | 169 | 39.8 +/- 2.7 | 35.1 +/- 3.0 | -4.7 |
| gold = positive | 341 | 86.7 +/- 1.0 | 89.4 +/- 2.5 | +2.7 |
| long sentence (>25 words) | 77 | 60.6 +/- 5.2 | 59.7 +/- 1.3 | -0.9 |
| mixed polarity in sentence | 103 | 44.7 +/- 4.4 | 45.0 +/- 5.3 | +0.3 |
| multi-aspect sentence | 379 | 72.4 +/- 1.9 | 73.1 +/- 1.6 | +0.7 |
| negation | 140 | 62.9 +/- 2.6 | 62.4 +/- 2.7 | -0.5 |

GCN right on all seeds and BERT wrong on all seeds: **1** examples; the reverse: **2**.

**GCN right, BERT wrong** (first 1):

| Sentence | Aspect | Gold | Wrong prediction | Tags |
|---|---|---|---|---|
| This version of MacBook Pro runs on a third-generation CPU ("Ivy Bridge"), not the latest fourth-generation Haswell CPU the 2013 version has. | third-generation CPU ("Ivy Bridge") | neutral | positive | multi-aspect sentence, negation |

**BERT right, GCN wrong** (first 2):

| Sentence | Aspect | Gold | Wrong prediction | Tags |
|---|---|---|---|---|
| Though please note that sometimes it crashes, and the sound quality isnt superb. | sound quality | negative | positive | contrast word |
| I understand I should call Apple Tech Support about any variables(which is my purpose of writing this con) as variables could be a bigger future problem. | Apple Tech Support | neutral | negative | long sentence (>25 words) |

## Train laptop -> test laptop (in-domain), seeds [42, 43, 44]

| Category | n | BERT acc | GCN acc | delta |
|---|---|---|---|---|
| all | 638 | 77.1 +/- 1.2 | 76.6 +/- 0.6 | -0.5 |
| contrast word | 76 | 61.8 +/- 4.6 | 63.2 +/- 0.0 | +1.3 |
| gold = negative | 128 | 87.0 +/- 1.8 | 85.4 +/- 1.6 | -1.6 |
| gold = neutral | 169 | 48.1 +/- 3.9 | 47.9 +/- 1.2 | -0.2 |
| gold = positive | 341 | 87.8 +/- 2.9 | 87.5 +/- 1.3 | -0.3 |
| long sentence (>25 words) | 77 | 67.5 +/- 3.4 | 68.8 +/- 2.2 | +1.3 |
| mixed polarity in sentence | 103 | 52.1 +/- 1.1 | 54.0 +/- 2.4 | +1.9 |
| multi-aspect sentence | 379 | 77.3 +/- 1.9 | 75.8 +/- 1.2 | -1.5 |
| negation | 140 | 62.6 +/- 0.4 | 63.6 +/- 1.4 | +1.0 |

GCN right on all seeds and BERT wrong on all seeds: **1** examples; the reverse: **3**.

**GCN right, BERT wrong** (first 1):

| Sentence | Aspect | Gold | Wrong prediction | Tags |
|---|---|---|---|---|
| Cost is more as compared to other brands. | Cost | negative | positive | - |

**BERT right, GCN wrong** (first 3):

| Sentence | Aspect | Gold | Wrong prediction | Tags |
|---|---|---|---|---|
| Though please note that sometimes it crashes, and the sound quality isnt superb. | sound quality | negative | positive | contrast word |
| I was given a demonstration of Windows 8. | Windows 8 | neutral | positive | - |
| At home and the office it gets plugged into an external 24" LCD screen, so built in screen size is not terribly important. | built in screen size | neutral | positive | multi-aspect sentence, negation |

