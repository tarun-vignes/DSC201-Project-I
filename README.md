# DSC 201 Project I: Edmunds Car Recommender System

This project uses 226,931 Edmunds consumer car reviews to build and compare three
recommendation systems:

1. A content-based model using TF-IDF review features and cosine similarity.
2. A collaborative recommender using sparse, user-centered SVD.
3. A hybrid model combining content similarity with personalized SVD predictions.

The fully executed notebook is available at
[`notebooks/Project_1_Edmunds_Car_Recommender_Tarun.ipynb`](notebooks/Project_1_Edmunds_Car_Recommender_Tarun.ipynb).

## Key results

- 226,931 raw reviews and 224,444 cleaned reviews
- 18,287 vehicles in the content catalog
- 70.3% same-manufacturer precision@10 for content recommendations
- 65.2% within-three-model-years precision@10
- 26,662 supported collaborative interactions
- 2,378 collaborative users and 3,384 collaborative vehicles
- 0.9647 SVD test RMSE and 0.6933 test MAE

The content model produced the strongest interpretable similarity results. The SVD
model supplied personalized rankings, although extreme rating sparsity prevented it
from outperforming the global-mean RMSE baseline. The hybrid model balanced vehicle
similarity and learned user preference.

## Dataset

Download the public
[Structured Edmunds Car Reviews Dataset](https://www.kaggle.com/datasets/mahisamy/structured-edmunds-car-reviews-dataset)
and save its archive as:

```text
data/structured_edmunds_car_reviews.zip
```

The raw dataset and local virtual environment are intentionally excluded from GitHub.
The notebook can read the extracted manufacturer CSVs from `data/raw/`, or it can
extract the archive automatically.

For Google Colab, place the archive or extracted `raw` folder under:

```text
My Drive/DSC201/Fall2026/Project_1/data/
```

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python build_notebook.py
python execute_notebook.py
```

## Repository contents

```text
notebooks/             Completed notebook with saved outputs
outputs/               Small reproducible result tables
build_notebook.py      Rebuilds the notebook source
execute_notebook.py    Executes and saves all notebook outputs
profile_raw_data.py    Profiles all raw CSV files using the standard library
requirements.txt       Python dependencies
```

## Required recommender interfaces

- `content_based_recommender(title, n)`
- `collaborative_recommender(user_id, n)`
- `hybrid_recommender(title, user_id, k, ...)`
