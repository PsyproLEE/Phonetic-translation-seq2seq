import pandas as pd
import random
import re
from noising.noise_generator import phonological_process

# Noise functions are conceptually based on KoNoise.
# See: https://github.com/wisenut-research/konoise

def phonological_noise(text, prob):
    return phonological_process(text, prob=prob)

def random_drop_noise(text, prob=0.2):
    return "".join([c for c in text if random.random() > prob])

def random_change_noise(text, prob=0.2):
    return re.sub("아", "야", text) if random.random() < prob else text

def augment_x5(df):
    augmented = []

    augmented.append(df)  # original
    augmented.append(df.assign(input=df["input"].apply(lambda x: phonological_noise(x, 0.3))))
    augmented.append(df.assign(input=df["input"].apply(lambda x: phonological_noise(x, 0.7))))
    augmented.append(df.assign(input=df["input"].apply(random_drop_noise)))
    augmented.append(df.assign(input=df["input"].apply(random_change_noise)))

    return pd.concat(augmented, ignore_index=True)