# from features.base import Feature
# from features.base import Feature, generate_features, create_memo

# from src.pre_fun import base_data

# import cudf
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from rdkit import Chem
from pathlib import Path
from mordred import Calculator, descriptors

import os


# from mordred import Calculator, descriptors


# Feature.dir = "../features_data"
# data = pd.read_csv("../datasets/dataset.csv")


def mordred_fe(data, cwd):
    # print(os.getcwd())
    filepath = cwd / "../features/mordred_fe.pkl"
    if not os.path.isfile(filepath):
        data["SMILES"] = data["SMILES"].transform(
            lambda x: Chem.MolFromSmiles(x)
        )
        calc = Calculator(descriptors, ignore_3D=True)

        new_data = calc.pandas(data["SMILES"])

        if cwd != Path(""):
            new_data.to_pickle(filepath)
    else:
        new_data = pd.read_pickle(filepath)

    return new_data


def fe(data, cwd):
    data["one_count_2"] = data["SMILES"].transform(lambda x: x.count("1")) == 2

    a = mordred_fe(data, cwd)
    data = pd.concat(
        [
            data,
            a
        ],
        axis=1
    )

    # カラム名は違えど要素が一緒のカラムは100個くらいあるけど気にしない（実行時間が長くなるので）
    # data = data.T.drop_duplicates().T
    data = data.loc[:, ~data.columns.duplicated()]

    data = data.drop(
        columns=["SMILES"]
    )
    return data


def run(cwd, data=False):
    if type(cwd) == str:
        cwd = Path(cwd)

    train = False
    if type(data) == bool:
        train = True
        data = pd.read_csv(cwd / "../datasets/dataset.csv")
        data = data.rename(
            columns={
                "log P (octanol-water)": "target"
            }
        )
    data = fe(data, cwd)

    if train:
        data.to_pickle(cwd / "../features/data_1.pkl")

    return data.astype(float)


if __name__ == "__main__":
    run(Path(""))
