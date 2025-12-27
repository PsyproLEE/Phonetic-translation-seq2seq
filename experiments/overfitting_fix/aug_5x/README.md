# Overfitting Mitigation via Multi-level Data Augmentation (x5)

This experiment aims to mitigate overfitting observed in baseline models
by expanding the training data using multiple levels of phonetic noise augmentation.

## Motivation

In baseline experiments, the model exhibited the following behavior:

- Training BLEU score increased rapidly
- Validation BLEU score plateaued or degraded after a certain number of epochs

This suggests that the model was overfitting to relatively clean and regular
phonetic input patterns, with insufficient input diversity.

## Augmentation Strategy

Starting from the original dataset, four additional augmented datasets were created
using different noise strategies, resulting in a total **5× expanded dataset**.

### Applied Noise Types

- Pronunciation noise (probability = 0.3)
- Pronunciation noise (probability = 0.7)
- Random drop noise  
  (random deletion of syllables or characters)
- Random change noise  
  (random substitution between phonetically similar characters)

All noise transformations are applied **only to the input (phonetic) sequences**.
Target (semantic) sentences remain unchanged.

## Dataset Composition
```
Total dataset =
  Original
  + Augmentation (prob=0.3)
  + Augmentation (prob=0.7)
  + Random drop noise
  + Random change noise
```


## Training Setup

- Identical Seq2Seq with Attention architecture as baseline
- Same optimizer and learning rate settings
- Model structure and training strategy are kept constant
  to isolate the effect of data augmentation

Detailed hyperparameters are specified in `config.yaml`.

## Expected Effect

The goal of this experiment is not merely to maximize BLEU scores, but to:

- Stabilize validation performance
- Improve generalization ability
- Reduce excessive memorization of training data


