from pathlib import Path

from pandas import read_csv

from category_production.category_production import CategoryProduction
from constants import SAVE_DIR

with Path(SAVE_DIR, "smd20.csv").open("r") as smd_file:
    smd20 = read_csv(smd_file, header=0, index_col=None)
with Path(SAVE_DIR, "ldd20.csv").open("r") as smd_file:
    ldd20 = read_csv(smd_file, header=0, index_col=None)

cp = CategoryProduction()

cp_data = cp.data


smd20_category = smd20.rename(columns={"Word": "SM_category",
                                       "SMD20 (Euclidean)": "Category SMD20 (Euclidean)",
                                       "SMD20 (cosine)": "Category SMD20 (cosine)",
                                       "SMD20 (correlation)": "Category SMD20 (correlation)",
                                       "SMD20 (Minkowski-3)": "Category SMD20 (Minkowski-3)",
                                       })
smd20_response = smd20.rename(columns={"Word": "SM_term",
                                       "SMD20 (Euclidean)": "Response SMD20 (Euclidean)",
                                       "SMD20 (cosine)": "Response SMD20 (cosine)",
                                       "SMD20 (correlation)": "Response SMD20 (correlation)",
                                       "SMD20 (Minkowski-3)": "Response SMD20 (Minkowski-3)",
                                       })

cp_data = cp_data.merge(smd20_category, on="SM_category", how="left")
cp_data = cp_data.merge(smd20_response, on="SM_term", how="left")

cp_data.to_csv(f"{SAVE_DIR}/Notes/2021-02-15 CP regression/cp.csv")

pass
