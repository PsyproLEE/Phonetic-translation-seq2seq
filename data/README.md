### Dataset Composition

- Base corpus: ~260K Japanese sentences (sentence-level)
- Dialogue data: used only for data augmentation / diversity
- Final training set: sentence-level (input, target) pairs

### Data Collection Note

The Papago crawling script is provided for documentation purposes only.
Due to ethical and legal considerations, the script is not intended to be
executed directly.

The final dataset was constructed prior to this repository and is included
in processed form.

### Data Structure

- raw/: original Japanese corpus
- processed/: final (input, target) pairs
- augmentation/: noise augmentation scripts
- splits/: train/val/test CSV files

Note: Crawling scripts are for documentation only.
