from sklearn.model_selection import KFold

import numpy as np
import pandas as pd
import hydra
import lightgbm as lgb
import gc
import os
import glob
import os.path
import zipfile
import joblib

from pathlib import Path

import mlflow
import mlflow.lightgbm

from src.preprocess import run
from utils import git_commits

rand = np.random.randint(0, 1000000)


def save_log(score_dict):
    mlflow.log_metrics(score_dict)
    mlflow.log_artifact(".hydra/config.yaml")
    mlflow.log_artifact(".hydra/hydra.yaml")
    mlflow.log_artifact(".hydra/overrides.yaml")
    mlflow.log_artifact(f"{os.path.basename(__file__)[:-3]}.log")
    mlflow.log_artifact("features.csv")


def add_all(zip_, files, arcnames):
    for file, arcname in zip(files, arcnames):
        if os.path.isfile(file):
            print(file, arcname)
            zip_.write(file, arcname=arcname)
            # print('  ', file)


def for_submit(cwd):
    file_name = cwd / f"../outputs/{rand}.zip"

    with zipfile.ZipFile(file_name, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        add_all(zipf, glob.glob(".hydra/config.yaml"), ["config.yaml"])
        add_all(zipf, glob.glob(str(cwd / "../env.yaml")),
                [os.path.basename(p.rstrip(os.sep)) for p in glob.glob(str(cwd / "../env.yaml"))])
        add_all(zipf, glob.glob(str(cwd / "**.py")),
                ["src/" + os.path.basename(p.rstrip(os.sep)) for p in glob.glob(str(cwd / "**.py"))])
        add_all(zipf, glob.glob(str(cwd / "../models/**.pkl")),
                [os.path.basename(p.rstrip(os.sep)) for p in glob.glob(str(cwd / "../models/**.pkl"))])


@git_commits(rand)
def main(cfg):
    cwd = Path(hydra.utils.get_original_cwd())

    data = run(cwd)

    train = data.copy()
    target = data["target"]
    train = train.drop(columns="target")

    kfold = KFold(
        n_splits=cfg.base.n_folds,
        shuffle=True,
        random_state=cfg.base.seed
    )

    print("file:///" + hydra.utils.get_original_cwd() + "mlruns")
    mlflow.set_tracking_uri("file://" + hydra.utils.get_original_cwd() + "/mlruns")

    use_cols = pd.Series(train.columns)
    use_cols.to_csv("features.csv", index=False, header=False)

    score = 0
    mlflow.lightgbm.autolog()
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(train, target)):
        x_train, x_valid = train.loc[train_idx], train.loc[valid_idx]
        y_train, y_valid = target[train_idx], target[valid_idx]

        d_train = lgb.Dataset(x_train, label=y_train)
        d_valid = lgb.Dataset(x_valid, label=y_valid)
        del x_train
        del x_valid
        del y_train
        del y_valid
        gc.collect()
        mlflow.set_experiment(f"fold_{fold + 1}")

        with mlflow.start_run(run_name=f"{rand}"):
            estimator = lgb.train(
                params=dict(cfg.parameters),
                train_set=d_train,
                num_boost_round=cfg.base.num_boost_round,
                valid_sets=[d_train, d_valid],
                verbose_eval=500,
                early_stopping_rounds=100
            )

            if cfg.base.submit:
                joblib.dump(
                    estimator,
                    cwd / f"../models/{fold + 1}.pkl"
                )

            print(fold + 1, "done")

            score_ = estimator.best_score["valid_1"][cfg.parameters.metric]
            score += score_ / cfg.base.n_folds

            save_log(
                {
                    "score": score
                }
            )

    for_submit(cwd)


@hydra.main(config_name="config/training.yaml")
def _run(cfg):
    main(cfg)


_run()
