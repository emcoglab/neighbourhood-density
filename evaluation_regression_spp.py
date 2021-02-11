from pathlib import Path

from pandas import read_csv

from ldm.evaluation.regression import SppData
from constants import SAVE_DIR

with Path(SAVE_DIR, "smd20.csv").open("r") as smd_file:
    smd20 = read_csv(smd_file, header=0, index_col=None)
with Path(SAVE_DIR, "ldd20.csv").open("r") as smd_file:
    ldd20 = read_csv(smd_file, header=0, index_col=None)
spp = SppData().dataframe
smd20_prime = smd20.rename(columns={"Word": "PrimeWord",
                                    # "SMD20 (Euclidean)": "Prime SMD20",
                                    # "SMD20 (cosine)": "Prime SMD20",
                                    # "SMD20 (correlation)": "Prime SMD20",
                                    "SMD20 (Minkowski-3)": "Prime SMD20",
                                    })
smd20_target = smd20.rename(columns={"Word": "TargetWord",
                                     # "SMD20 (Euclidean)": "Target SMD20",
                                     # "SMD20 (cosine)": "Target SMD20",
                                     # "SMD20 (correlation)": "Target SMD20",
                                     "SMD20 (Minkowski-3)": "Target SMD20",
                                     })
ldd20_prime = ldd20.rename(columns={"Word": "PrimeWord",
                                    "LDD20 (PPMI n-gram (BBC), r=5)": "Prime LDD20",
                                    })
ldd20_target = ldd20.rename(columns={"Word": "TargetWord",
                                     "LDD20 (PPMI n-gram (BBC), r=5)": "Target LDD20",
                                     })

spp = spp.merge(smd20_prime[["PrimeWord", "Prime SMD20"]], on="PrimeWord", how="left")
spp = spp.merge(smd20_target[["TargetWord", "Target SMD20"]], on="TargetWord", how="left")
spp = spp.merge(ldd20_prime[["PrimeWord", "Prime LDD20"]], on="PrimeWord", how="left")
spp = spp.merge(ldd20_target[["TargetWord", "Target LDD20"]], on="TargetWord", how="left")

spp.to_csv(f"{SAVE_DIR}/Notes/2021-02-11 SPP regression/spp.csv")

pass
