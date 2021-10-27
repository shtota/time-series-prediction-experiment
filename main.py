import os
import pandas as pd
import numpy as np

import pytorch_lightning as pl

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
df = pd.read_csv('./data/Binance_BTCUSDT_1h.csv', parse_dates=['date'])
df = df[df.date > "2018-01-01"]
#for f in os.listdir('./data/'):
#    tmp = pd.read_csv('./data/'+f, parse_dates=['date'])
#    df[f[8:11]] = tmp.set_index('date').loc[df.date.values].open.values
#df = pd.concat(df)
df["time_idx"] = df.shape[0] - df.index.values
# define dataset
max_encoder_length = 36
max_prediction_length = 6
training_cutoff = "2020-05-01"  # day for cutoff

training = TimeSeriesDataSet(
    df[df.date < training_cutoff],
    time_idx='time_idx',
    target='open',
    min_prediction_idx=10000,
    # weight="weight",
    group_ids=['symbol'],
    max_encoder_length=36,
    max_prediction_length=6,
    time_varying_unknown_reals=['open'],
)

print('win')

pl.