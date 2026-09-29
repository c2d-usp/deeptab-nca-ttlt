# On the Role of Neighborhood Purity in Local Test-Time Adaptation for Tabular Data

This is the official implementation of the paper "On the Role of Neighborhood Purity in Local Test-Time Adaptation for Tabular Data" (SIBGRAPI 2026).

Test-Time Local Training (TTLT) fine-tunes a pre-trained network on the *k* nearest training neighbors of each query before predicting it.
We propose **NCA-TTLT**, which retrieves those neighbors in a supervised metric space learned by
[Neighbourhood Component Analysis (NCA)](https://www.cs.toronto.edu/~hinton/absps/nca.pdf) instead of the original feature space.
The retrieved support sets are purer (more same-class neighbors), which makes query-specific adaptation more effective.

Table of Contents:
- [The main results](#the-main-results)
- [How to reproduce the results](#how-to-reproduce-the-results)
    - [Set up the environment](#set-up-the-environment)
        - [Software](#software)
        - [Data](#data)
    - [Quick test](#quick-test)
    - [Tutorial](#tutorial)
    - [Reproducing other results](#reproducing-other-results)
- [Understanding the repository](#understanding-the-repository)
    - [Code overview](#code-overview)
    - [Running scripts](#running-scripts)
    - [Technical notes](#technical-notes)
- [Adding new datasets](#adding-new-datasets)
- [How to cite](#how-to-cite)

# The main results

Test accuracy in % (mean ± std over 15 seeds) on the eight classification datasets from the paper.
k-NN scenarios do not depend on the seed; all scenarios use *k* = 10; TTLT and NCA-TTLT differ **only** in the space used to retrieve the neighborhood.

| ID | Dataset | Original k-NN | NCA k-NN | Global MLP | MLP + TTLT | MLP + NCA-TTLT (ours) |
|----|---------|:---:|:---:|:---:|:---:|:---:|
| AD | adult                             | 83.2 | 84.6 | 85.2 ± 0.1 | 85.1 ± 0.1 | 85.2 ± 0.1 |
| BC | Bank Customer Churn Dataset       | 83.4 | 84.6 | 86.1 ± 0.2 | 86.2 ± 0.3 | 85.9 ± 0.2 |
| CH | California-Housing-Classification | 84.4 | 88.9 | 88.6 ± 0.2 | 88.5 ± 0.2 | **88.8 ± 0.2** |
| EL | electricity                       | 82.0 | 83.2 | 84.3 ± 0.4 | 84.6 ± 0.3 | 84.6 ± 0.3 |
| HO | house_16H                         | 84.4 | 85.6 | 86.4 ± 0.5 | 86.4 ± 0.3 | 86.4 ± 0.4 |
| WI | wine                              | 70.6 | 71.0 | 73.8 ± 0.8 | 74.0 ± 0.6 | 74.0 ± 0.7 |
| CA | car-evaluation                    | 87.0 | 92.8 | 93.4 ± 3.3 | 93.4 ± 3.3 | 93.4 ± 3.3 |
| LE | letter                            | 92.9 | 95.9 | 96.2 ± 0.5 | 96.6 ± 0.5 | **96.7 ± 0.5** |

See the paper for the neighborhood-purity analysis (Figures 1 and 2) and the paired statistical tests.

# How to reproduce the results

## Set up the environment

### Software

The code was developed with Python 3.14 and PyTorch 2.14 (CUDA 13.0). A plain `venv` + `pip` is enough:

```shell
git clone https://github.com/c2d-usp/deeptab-nca-ttlt
cd deeptab-nca-ttlt
python -m venv .venv
source .venv/bin/activate            # on Windows: .venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

`requirements.txt` pins the CUDA 13.0 build of PyTorch (`torch==2.14.0+cu130`), which is why the extra index URL is needed.
If your machine does not have a GPU, or has a different CUDA version, install the matching PyTorch build first
([instructions](https://pytorch.org/get-started/locally/)) and then install the remaining packages.
The code picks the GPU automatically when one is available and falls back to the CPU otherwise.

### Data

(***License:** we do not impose any new license restrictions in addition to the original licenses of the used datasets.
The `source` field of each `data/<dataset>/info.json` points to the original source of the dataset.*)

All datasets in the `data/` directory come from the public benchmark of the
**[TALENT](https://github.com/LAMDA-Tabular/TALENT)** toolbox ("A Tabular Analytics and Learning Toolbox", Liu et al., JMLR 2025).
We use the same fixed train/validation/test splits provided by TALENT, so no extra download or preprocessing step is required:
after cloning, `data/` already contains ~300 datasets in the TALENT format (see [Adding new datasets](#adding-new-datasets)).

The eight datasets used in the paper correspond to the following directories:

| ID | Directory | Task | Train | Val | Test |
|----|-----------|------|------:|----:|-----:|
| AD | `data/adult`                             | binclass   | 26,048 | 6,513 | 16,281 |
| BC | `data/Bank_Customer_Churn_Dataset`       | binclass   |  6,400 | 1,600 |  2,000 |
| CH | `data/California-Housing-Classification` | binclass   | 13,209 | 3,303 |  4,128 |
| EL | `data/electricity`                       | binclass   | 28,999 | 7,250 |  9,063 |
| HO | `data/house_16H`                         | binclass   |  8,632 | 2,158 |  2,698 |
| WI | `data/wine`                              | binclass   |  1,634 |   409 |    511 |
| CA | `data/car-evaluation`                    | multiclass |  1,105 |   277 |    346 |
| LE | `data/letter`                            | multiclass | 12,800 | 3,200 |  4,000 |

## Quick test

To check that the environment is configured correctly, run the full pipeline on the smallest dataset from the paper
(takes about 2-3 minutes on a laptop CPU):

```shell
python run.py --dataset wine
```

The last lines of the output log should look like this (the exact values vary between runs):

```
Model saved to output/wine/model.pt
Metrics saved to output/wine/metrics.json
{
  "accuracy": 0.73...,
  "f1_macro": 0.73...,
  "f1_weighted": 0.73...
}
```

## Tutorial

Here, we reproduce one run of the adaptation study (Global MLP vs. MLP + NCA-TTLT) on the Adult dataset (in the paper, this dataset is referred to as "AD").
*Running the other datasets is identical, only the `--dataset` argument changes.*

```shell
python run.py --dataset adult
```

`run.py` performs the following steps:
1. Loads `data/adult`, one-hot encodes the categorical features and standardizes all features using training statistics.
2. Trains the global MLP (2 hidden layers × 100 units, `tanh`) with Adam and early stopping on the validation split.
3. Evaluates the global MLP on the test split (**Global MLP** baseline).
4. Fits NCA on the training split and, for every test query, retrieves its *k* = 10 nearest training samples **in the NCA space**,
   fine-tunes a copy of the global MLP on them for `T = 100` steps with a distance-weighted loss, and predicts the query (**MLP + NCA-TTLT**).
   Each query starts from the global MLP; adapted weights are never carried over to the next query.

Once the run is finished, the `output/adult/` directory should appear:
- `standard.json` — test metrics of the global MLP (accuracy, macro/weighted F1, per-class report)
- `metrics.json` — test metrics of MLP + NCA-TTLT (same keys)
- `model.pt` — `state_dict` of the global MLP

To run all eight datasets from the paper:

```shell
for d in adult Bank_Customer_Churn_Dataset California-Housing-Classification electricity house_16H wine car-evaluation letter; do
    python run.py --dataset "$d"
done
```

## Reproducing other results

**MLP + TTLT (original feature space).**
The original TTLT of Kang and Kang ("Test-time local training of neural network for tabular data", *Scientific Reports*, 2026) is the same procedure without the NCA projection.
To run it, pass `projection=None` to `ttlt_prediction` in `run.py` (the NCA fit right above the call is then no longer needed):

```python
pred = ttlt_prediction(
    X_test_t[i], X_train_t, y_train_t,
    model=model,
    projection=None,   # original TTLT: neighbors retrieved with Euclidean distance in the input space
    task=task,
    k=(100 if task == "regression" else 10),
    T=100
)
```

**Multiple seeds.** The paper reports the mean ± std over 15 paired seeds. `run.py` performs a single run and does not fix the random seed,
so repeated runs give slightly different numbers.


# Understanding the repository

Read this if you are going to do more experiments/research in this repository.

## Code overview
- `run.py` — end-to-end experiment for one dataset: loading, preprocessing, global MLP training (`train_model`), standard inference,
  NCA-TTLT inference, metrics (`compute_metrics`) and saving the results
- `ttlt.py` — the test-time local training method
    - `get_k_nearest_neighbors` retrieves the *k* nearest training samples, either in the input space (`projection=None`)
      or in the space of a fitted projection with a scikit-learn-style `transform` method (e.g. `NeighborhoodComponentsAnalysis`)
    - `ttlt_prediction` adapts a copy of the model to the retrieved neighborhood and predicts a single query.
      Note that the neighbors are *retrieved* in the projected space but the model is *fine-tuned* on their original features.
- `model_mlp.py` — the MLP backbone used for global training
- `data_val.py` — `get_dataset`, the loader for the TALENT data format
- `data/` — the TALENT benchmark datasets (see [Data](#data))

## Running scripts

`run.py` takes a single argument, the name of a directory in `data/`:

```shell
python run.py --dataset <dataset name>
```

The results are written to `output/<dataset name>/`. Running the same dataset again **overwrites** the existing results.

The task is inferred from the `task_type` field of `info.json`: `binclass` and `multiclass` are treated as classification,
everything else as regression. For regression, targets are standardized for training, predictions are mapped back to the original scale,
and `metrics.json`/`standard.json` report RMSE, MAE and R².

## Technical notes
- The hyperparameters are defined directly in the code:

    | Stage | Hyperparameter | Value | Where |
    |-------|----------------|-------|-------|
    | Global MLP | architecture | 2 × 100, `tanh` | `run.py` / `model_mlp.py` |
    | Global MLP | optimizer | Adam, lr = 1e-3, weight decay = 1e-5 | `train_model` |
    | Global MLP | batch size / max epochs | 50 / 500 | `train_model` |
    | Global MLP | early stopping patience | 100 epochs (+ `ReduceLROnPlateau`, factor 0.1, patience 50) | `train_model` |
    | NCA | initialization | identity, `random_state=42`, no dimensionality reduction | `run.py` |
    | TTLT | neighbors *k* | 10 (classification), 100 (regression) | `run.py` |
    | TTLT | local steps *T* | 100 | `run.py` |
    | TTLT | optimizer | Adam, lr = 1e-5, weight decay = 1e-5 | `ttlt_prediction` |
    | TTLT | neighbor weights | inverse distance, normalized to sum to 1 | `ttlt_prediction` |

- NCA is fit **only** on the training split; validation and test samples are projected with the learned mapping, never refit.
- Categorical features are one-hot encoded (unknown categories are ignored) and concatenated to the numerical features before standardization,
  so NCA and the MLP operate on the same input representation.
- TTLT inference processes one query at a time and fits NCA on the full training set. On large datasets (e.g. `adult`, `electricity`), this is the
  most expensive part of the pipeline; a GPU helps with the local fine-tuning, but NCA itself runs on the CPU (scikit-learn).

# Adding new datasets

To use your own dataset, create a new directory in `data/` with **the same file names** as the TALENT datasets.
A good example is `data/adult`, which has both numerical and categorical features.
The `.npy` files are NumPy arrays saved with the `np.save` function ([documentation](https://numpy.org/doc/stable/reference/generated/numpy.save.html)).

Let's say your dataset is called `my-dataset`. Then, create the `data/my-dataset` directory with the following content:
- If the dataset has numerical features
    - Files: `N_train.npy`, `N_val.npy`, `N_test.npy`
    - Shape: `(n_samples, n_num_features)`, float values
- If the dataset has categorical features
    - Files: `C_train.npy`, `C_val.npy`, `C_test.npy`
    - Shape: `(n_samples, n_cat_features)`; values are one-hot encoded by `get_dataset`, so strings are fine
- At least one of the two groups above must be present.
- Labels
    - Files: `y_train.npy`, `y_val.npy`, `y_test.npy`
    - Shape: `(n_samples,)`. For classification, labels are re-encoded with `LabelEncoder`, so any label values work.
- `info.json` — a JSON file with at least the following key:
    - `"task_type"`: one of `"regression"`, `"binclass"`, `"multiclass"`

Then run:

```shell
python run.py --dataset my-dataset
```

# How to cite

```
@inproceedings{cunha2026neighborhood,
    title={On the Role of Neighborhood Purity in Local Test-Time Adaptation for Tabular Data},
    author={
        Victor Cunha and
        Bruno Maia and
        Beatriz Buzatto and
        Erick Sousa and
        Thiago Mantovan and
        Tiago Nazar{\'e} and
        Jones Egydio and
        Anna Helena Reali Costa and
        Artur Jord{\~a}o
    },
    booktitle={Conference on Graphics, Patterns and Images (SIBGRAPI)},
    year={2026},
}
```

This work builds on the TALENT benchmark. If you use the data from this repository, please also cite:

```
@article{liu2025talent,
    title={TALENT: A Tabular Analytics and Learning Toolbox},
    author={
        Liu, S.-Y. and
        Cai, H.-R. and
        Zhou, Q.-L. and
        Yin, H.-H. and
        Zhou, T. and
        Jiang, J.-P. and
        Ye, H.-J.
    },
    journal={Journal of Machine Learning Research},
    year={2025},
}
```
