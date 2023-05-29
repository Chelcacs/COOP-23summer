# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Downloads and prepares the Forest Covertype dataset."""

import gzip
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import wget



def main():
  
  if not os.path.exists('./data'):
    os.makedirs('./data')

  df = pd.read_csv('../data_preprocess/standardized_data.csv')
  n_total = len(df)

  # Train, val and test split follows

  train_val_indices, test_indices = train_test_split(
      range(n_total), test_size=0.2, random_state=0)
  train_indices, val_indices = train_test_split(
      train_val_indices, test_size=0.2 / 0.6, random_state=0)

  traindf = df.iloc[train_indices]
  valdf = df.iloc[val_indices]
  testdf = df.iloc[test_indices]
  traindf = traindf.sample(frac=1)

  traindf.to_csv('data/train_covertype.csv', index=False, header=False)
  valdf.to_csv('data/val_covertype.csv', index=False, header=False)
  testdf.to_csv('data/test_covertype.csv', index=False, header=False)

if __name__ == '__main__':
  main()
