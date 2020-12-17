import sys
import joblib
import pandas as pd
import numpy as np
# import deepchem
import lightgbm
# from preprocess import run
import yaml
import os

debug = False
if debug:
    from preprocess import run
else:
    from preprocess import run

if not debug:
    input_data = []
    for line in sys.stdin:
        input_data.append(line.strip().split(","))

    input_df = pd.DataFrame(data=input_data, columns=["SMILES"])
    data = input_df.replace("", None)
else:

    input_data = []
    for line in sys.stdin:
        input_data.append(line.strip().split(","))

    input_df = pd.DataFrame(data=input_data, columns=["SMILES"])
    data = input_df.replace("", None)  # .drop(columns="log P (octanol-water)")
# print(input_data, data.shape)
data = run("", data, debug).astype(float)
# print(os.getcwd())
# print(data.shape)
filename = "config/training.yaml" if debug else "config.yaml"
with open(filename, "r+") as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)

pred = np.zeros(data.shape[0])
for fold in range(1, cfg["base"]["n_folds"] + 1):
    path = f"../models/{fold}.pkl" if debug else f"{fold}.pkl"
    estimator = joblib.load(path)

    pred += estimator.predict(data) / cfg["base"]["n_folds"]

for val in pred:
    val = float(val)
    print(val)
    # print(type(val))
