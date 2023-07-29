import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load CSV file
data_2 = pd.read_csv("heart_disease_uci.csv")

#data preprocessing for dataset 2
data_2.drop(['id','dataset'], axis=1, inplace=True)

data_2['sex'].replace('Male',1,inplace=True)
data_2['sex'].replace('Female','0',inplace=True)

data_2['cp'].replace('typical angina',1,inplace=True)
data_2['cp'].replace('atypical angina',2,inplace=True)
data_2['cp'].replace('non-anginal',3,inplace=True)
data_2['cp'].replace('asymptomatic',4,inplace=True)


data_2['fbs'].replace('TRUE',1,inplace=True)
data_2['fbs'].replace('FALSE',0,inplace=True)

data_2['restecg'].replace('normal',0,inplace=True)
data_2['restecg'].replace('st-t abnormality',1,inplace=True)
data_2['restecg'].replace('lv hypertrophy',2,inplace=True)

data_2['exang'].replace('TRUE',1,inplace=True)
data_2['exang'].replace('FALSE',0,inplace=True)

data_2['slope'].replace('upsloping',1,inplace=True)
data_2['slope'].replace('flat',2,inplace=True)
data_2['slope'].replace('downsloping',3,inplace=True)

data_2['thal'].replace('normal',1,inplace=True)
data_2['thal'].replace('fixed defect',2,inplace=True)
data_2['thal'].replace('reversable defect',3,inplace=True)

data_2['num'].replace(['2','3','4'],1,inplace=True)


data_2.rename(columns={'num':'target'}, inplace=True)

# Specify the columns to standardize and normalize
columns_to_standardize = [
    "cp", "fbs", "restecg", 
    "exang", "slope", "thal"
]
# columns_to_normalize = ["height", "weight"]

# Standardize the specified columns
scaler = StandardScaler()
data_2[columns_to_standardize] = scaler.fit_transform(data_2[columns_to_standardize])

# Normalize the specified columns
# normalizer = MinMaxScaler()
# df[columns_to_normalize] = normalizer.fit_transform(df[columns_to_normalize])

# Save the standardized and normalized dataset to a CSV file
data_2.to_csv("standardized_uci.csv", index=False)