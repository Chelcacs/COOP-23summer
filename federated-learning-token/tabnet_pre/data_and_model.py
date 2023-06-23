from pytorch_tabnet.tab_model import TabNetClassifier

import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np
np.random.seed(0)


import os
import wget
from pathlib import Path

from matplotlib import pyplot as plt

import os
os.environ['CUDA_VISIBLE_DEVICES'] = f"1"

import torch
torch.__version__

#dataset loading and spliting
dataset_name = 'HEART DISEASE DATASET (COMPREHENSIVE)'
train = pd.read_csv("/Users/a123/Desktop/coop/COOP-23summer/federated-learning-token/tabnet_pre/standardized_data.csv")

target = 'target'
if "Set" not in train.columns:
    train["Set"] = np.random.choice(["train", "valid", "test"], p =[.8, .1, .1], size=(train.shape[0],))

train_indices = train[train.Set=="train"].index
valid_indices = train[train.Set=="valid"].index
test_indices = train[train.Set=="test"].index


#simple processing
nunique = train.nunique()
types = train.dtypes

categorical_columns = []
categorical_dims =  {}
for col in train.columns:
    if types[col] == 'object' or nunique[col] < 200:
        print(col, train[col].nunique())
        l_enc = LabelEncoder()
        train[col] = train[col].fillna("VV_likely")
        train[col] = l_enc.fit_transform(train[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)
    else:
        train.fillna(train.loc[train_indices, col].mean(), inplace=True)


#Define categorical features for categorical embeddings
unused_feat = ['Set']

features = [ col for col in train.columns if col not in unused_feat+[target]] 

cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]


# Network parameters
tabnet_params = {"cat_idxs":cat_idxs,
                 "cat_dims":cat_dims,
                 "cat_emb_dim":2,
                 "optimizer_fn":torch.optim.Adam,
                 "optimizer_params":dict(lr=2e-2),
                 "scheduler_params":{"step_size":50, # how to use learning rate scheduler
                                 "gamma":0.9},
                 "scheduler_fn":torch.optim.lr_scheduler.StepLR,
                 "mask_type":'entmax', # "sparsemax"
                #  "grouped_features" : grouped_features
                }

clf = TabNetClassifier(**tabnet_params
                      )

def get_train_data():
    X_train = train[features].values[train_indices]
    y_train = train[target].values[train_indices]
    return X_train, y_train
def get_val_data():
    X_valid = train[features].values[valid_indices]
    y_valid = train[target].values[valid_indices]
    return X_valid, y_valid
def get_test_data():
    X_test = train[features].values[test_indices]
    y_test = train[target].values[test_indices]
    return X_test, y_test
def get_model():
    return clf

   
