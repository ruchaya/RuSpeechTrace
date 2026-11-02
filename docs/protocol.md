# Frozen experimental protocol

1. Audit RuASD locally before training.
2. Use raw clean recordings for the main closed-set training.
3. A normalized text hash may belong to only one global split.
4. Augmented derivatives remain in the same global split as their source.
5. Main closed-set metric: Macro-F1.
6. Open-set evaluation uses five fixed generator-level folds.
7. Final withheld generators are never used to tune rejection thresholds.
8. Robustness starts from clean-trained frozen models.
9. Cross-dataset transfer uses only evidence-backed common generator families.
10. Model-level and family-level attribution are reported separately.
11. Final systems are repeated on at least five seeds.
