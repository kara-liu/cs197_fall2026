# Tabular VAE: a beginner-friendly starter kit (Jupyter version)

Simple Variational Autoencoder (VAE) for tabular data with **M categorical** and
**K continuous** columns. 

## Setup
First, make sure you are in the cs197 conda environment. Then install:
```bash
pip install torch
pip install wandb        # optional, only if you want experiment tracking
```

**Weights & Biases:** in the settings cell set `USE_WANDB = True`, and run `wandb login` once.

## What is in each file

| File | What it does |
|---|---|
| `tabular_vae.ipynb` | The main notebook: data, split, preprocessing, training loop, loss curves, synthetic data, exercises. |
| `encoders.py` | Only relevant for Project 3!! Turns one continuous variable into `m >= 1` dimensional encoding. For example, `StandardEncoder` (m = 1) and `ModeSpecificEncoder` (m = 1 + number of peaks). |
| `preprocessing.py` | `TabularPreprocessor`: one-hot encodes categoricals, applies one encoder per continuous column, builds the design matrix, and then can recover the original, raw values. |
| `vae.py` | `TabularVAE` (encoder, sampling, decoder) and `vae_loss`. |
| `scarf_model.py` | Only relevant for Project 4. SCARF model |

After `TabularPreprocessor`, each preprocessed row of patient data looks like this:

```
[ one-hot(cat_variable_1) | ... | one-hot(cat_variable_M) | encoding(cont_variable_1) | ... | enc(cont_variable_K) ]
```

## VAE loss

```
VAEloss = recon_cat + recon_cont + beta * KL
```

* `recon_cat`: reconstruction for categorical so cross-entropy, one per categorical variable, summed.
* `recon_cont`: reconstruction for continuous, squared error over the continuous part.
* `KL`: pulls the encoder's output q(z|x) toward N(0, 1), so we can sample new latents from N(0, 1) later.

The notebook logs all three terms (plus the total) for both the training and validation sets.

## Reading the curves

* Train and validation both fall: learning works.
* Train falls, validation rises: overfitting. Try a smaller `HIDDEN_DIM` or `LATENT_DIM`.
* KL close to 0: posterior collapse (the decoder ignores `z`). Try a smaller `BETA` or a longer `KL_WARMUP_EPOCHS`.

## Notes

* The preprocessor is fitted on the training split only, to avoid data leakage.
  Validation rows with categories unseen in training are dropped.
* Rows with missing values must be dropped or filled first.
* To use your own data, replace the `make_synthetic_data()` cell with `pd.read_csv(...)` and
  set `cat_cols` and `cont_cols`.

*This was generated with help from Claude.*