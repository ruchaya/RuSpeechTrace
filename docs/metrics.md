# Metric conventions

All OOD scoring functions return an ID-oriented score: larger = more likely known/ID.

- MSP: maximum softmax probability.
- Standard energy ID score: `T * logsumexp(logits/T)`.
- SME ID score: `T * logsumexp(softmax(logits/T))`, i.e. negative of the SME energy defined by Klein, Tak & Khoury (Interspeech 2025).
- FPR95: known/ID is the positive class; false positive rate on OOD at TPR >= 0.95.
- OSCR: area under Correct Classification Rate vs unknown False Positive Rate.
