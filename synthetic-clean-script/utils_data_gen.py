import numpy as np
import pandas as pd
import itertools

from numpy.random import choice
from numpy.random import binomial
from numpy.random import uniform as unif

from collections import defaultdict
from sklearn.linear_model import LogisticRegression


def make_preds(df, predictors, models):
    flt_str = {"S": "index==index", "A": "S==1", "Y0": "S==1 & A==0", "Y1": "S==1 & A==1"}

    for key, model in models.items():
        df[f"hat_P({key}=1)"] = model.predict_proba(df[predictors])[:,-1]
        df[f"SE_{key}"] = (df.query(flt_str[key])[key] - df[f"hat_P({key}=1)"]) ** 2


def merge_df_val(df_rct, df_obs, predictors, rct_models, obs_models):
    df = pd.concat([df_rct, df_obs]).reset_index(drop=True)

    pr_model = LogisticRegression().fit(df[predictors], df["R"])

    df["hat_P(R=1)"] = pr_model.predict_proba(df[predictors])[:,-1]
    df["SE_R"] = (df["R"] - df[f"hat_P(R=1)"]) ** 2

    df["mu_0_rct"] = rct_models["Y0"].predict_proba(df[predictors])[:,-1]
    df["mu_1_rct"] = rct_models["Y1"].predict_proba(df[predictors])[:,-1]
    df["mu_0_obs"] = obs_models["Y0"].predict_proba(df[predictors])[:,-1]
    df["mu_1_obs"] = obs_models["Y1"].predict_proba(df[predictors])[:,-1]

    df["b1(X)"] = (df["mu_1_rct"] - df["mu_0_rct"]) - (df["mu_1_obs"] - df["mu_0_obs"])
    df["abs(b1(X))"] = abs(df["b1(X)"])

    return df


def covs_to_prob(row, covs, p):
    group_index = 0
    for i, cov in enumerate(covs[::-1]):
        group_index += (2 ** i) * row[cov]

    return p[int(group_index)]


def sample_probs(d, pl_range, ph_range, bias_flag):
    p = defaultdict()
    for k in range(2 ** d):
        p[k] = choice([unif(*pl_range), unif(*ph_range)])

    p = list(p.values())
    if not bias_flag:
        p[1::2] = p[::2]

    return p


def sample_all_probs(d, pl_range, ph_range, scenario):
    probs = defaultdict(list)

    for key, bias_flag in scenario.items():
        probs[key] = sample_probs(d, pl_range, ph_range, bias_flag)

    return probs


def init_df(n, d, d_meas, r, probs, x_probs, trs_bias):
    covs = [f"X{i + 1}" for i in range(d)]
    meas_covs = [f"X{i + 1}" for i in range(d_meas)]
    
    X_meas = choice([0, 1], size=(n, d_meas), p=x_probs[f"R={r}"])

    for i, c in enumerate(list(itertools.product([0, 1], repeat=d_meas))):
        df[f"Xp{i + 1}"] = (df[meas_covs] == c).all(axis=1).astype(int)

    if trs_bias:
        u_prob = sample_probs(d_meas, (0.2, 0.8), (0.2, 0.8), True)
        df["P(U=1)"] = df.apply(lambda row: covs_to_prob(row, covs, u_prob), axis=1)
        df["U"] = df.apply(lambda row: binomial(1, row["P(U=1)"]), axis=1)
    else:
        U = choice([0, 1], size=(n,1), p=x_probs[f"R={r}"])

    X = np.concatenate((X_meas, U), axis=1)
    df = pd.DataFrame({**{'R': r}, **{cov: X[:,i] for i, cov in enumerate(covs)}})

    for key, prob in probs.items():
        df[f"P({key}=1)"] = df.apply(lambda row: covs_to_prob(row, covs, prob), axis=1)
        df[key] = df.apply(lambda row: binomial(1, row[f"P({key}=1)"]), axis=1)

    df["Y"] = df["A"] * df["Y1"] + (1 - df["A"]) * df["Y0"]   

    return df