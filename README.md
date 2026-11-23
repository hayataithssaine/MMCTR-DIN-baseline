# MMCTR-DIN-baseline
# MM-CTR Challenge : baseline DIN

Participation à la compétition **MM-CTR (EReL@MIR, Task 2)**.
Lien : https://www.codabench.org/competitions/5372/

## Modèle
- Baseline **DIN** (FuxiCTR 2.3.10) : https://github.com/reczoo/WWW2025_MMCTR_Challenge
- Dataset : MicroLens_1M_x2
- Hyperparamètres : embedding_dim = 32, embedding_regularizer = 1e-6,
  dnn_hidden_units = [1024, 512, 256], batch_size = 8192, lr = 1e-3
- Entraînement sur Kaggle (GPU T4), early stopping (patience = 3)

## Résultats
| | AUC |
|---|---|
| Validation | 0.7597 |
| Test (Codabench) | 0.7608 |

Soumission : `DIN_MicroLens_1M_x2_001_bb2da5f7.zip` (ID 1064479)
Classement : voir `leaderboard.png`

## Contenu du dépôt
- `notebook.ipynb` : code complet (installation, données, entraînement, prédiction)
- `leaderboard.png` : capture du classement
- `prediction.csv` : prédictions sur le test

## Reproduire
1. Cloner le dépôt de la compétition et installer `fuxictr==2.3.10`
2. Télécharger le dataset MicroLens_1M_x2 (ModelScope : reczoo/MicroLens_1M_x2)
3. `python run_param_tuner.py --config config/DIN_microlens_mmctr_tuner_config_01.yaml --gpu 0`
4. `python prediction.py --config config/DIN_microlens_mmctr_tuner_config_01 --expid DIN_MicroLens_1M_x2_001_bb2da5f7 --gpu 0`
