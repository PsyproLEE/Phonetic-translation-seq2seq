## Experiments

The following experiments evaluate the impact of noise intensity
and data scaling on translation robustness.


- augmentation-0.3: BLEU 0.5608
- augmentation-0.5: BLEU 0.5468
- augmentation-0.7: BLEU 0.5414
- augmentation-1.0: BLEU 0.5564
- augmentation-x5: BLEU 0.5523

These results indicate that moderate noise improves robustness,
while excessive noise intensity or scaling yields diminishing returns.

Noise intensity and data scaling were evaluated to mitigate
overfitting observed in the baseline experiments.
