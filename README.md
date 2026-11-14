# DSCI 552 — Homework 7

Chengfeng Tang · GitHub: tcf2021

[Homework notebook](notebook/Tang_Chengfeng_HW7.ipynb) contains the executed multi-output SVM and k-means experiments, the solution to ISLR 12.6.2, and the optional classifier-chain and classification-metric experiments.

| Path | Contents |
|---|---|
| `notebook/` | Executed notebook with question-by-question answers |
| `data/anuran/` | Original supplied MFCC CSV and Readme |
| `src/hw7_utils.py` | Model selection, multi-class classifier chain, and repeated clustering |
| `results/` | Row split, CV results, selected parameters, predictions, metrics, and plots |
| `results/kmeans_50_runs.csv` | Selected k and evaluation metrics for each repetition |
| `results/kmeans_all_candidates.csv` | All 2,500 k-means candidate fits |
| `results/kmeans_cluster_majorities.csv` | All selected clusters and their majority label triplets |

## Run

Tested with Python 3.12. From the repository root:

```sh
python -m pip install -r requirements.txt
python -m notebook notebook/Tang_Chengfeng_HW7.ipynb
```

Restart the kernel and run all cells. Paths work from the notebook folder or repository root. The helper uses three parallel workers for CV and clustering; a full run includes 50 clustering repetitions and can take several minutes.

## Experiment setup

The supervised experiments use a fixed random 70/30 row split and ten-fold stratified CV within training. RBF SVMs use explicit one-versus-rest classification. L1 SVMs standardize within each fold. The imbalance experiment uses inverse-frequency class weighting, one of the alternatives allowed in the assignment. RecordID is excluded from predictors; this row split does not measure generalization to unseen recordings.

Clustering uses all rows and the original normalized MFCC values. Each repetition selects k by the CH index over k = 2,...,50, with k = 1 retained as an undefined-CH baseline. Three initializations are used per candidate. Majority labels are assigned after k selection.

Hamming distance counts wrong entries in a three-label triplet. Hamming loss divides that count by three. Both label-accuracy Hamming score and example-based Jaccard score are reported to make the score convention explicit.

Data: [Anuran Calls (MFCCs), UCI](https://archive.ics.uci.edu/dataset/406/anuran+calls+mfccs), Colonna and colleagues. Source details are retained in `data/anuran/Readme.txt`.
