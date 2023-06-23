import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data_1 = df = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")
data_2 = df = pd.read_csv("heart_disease_uci.csv", dtype='str')


# data preprocessing for dataset 1
data_1.insert(11, 'ca', 0)
data_1.insert(12, 'thal', 0)

data_1.rename(columns={'chest pain type':'cp', 'resting bp s':'trestbps',
                       'cholesterol':'chol','fasting blood sugar':'fbs',
                       'resting ecg':'restecg','max heart rate':'thalch',
                       'exercise angina':'exang','ST slope':'slope',
                       'target':'num'}, inplace=True)
# data_1.to_csv("heart_statlog_cleveland_hungary_final_PROCESSED.csv", index=False)

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


# data_2.to_csv("heart_disease_uci_PROCESSED.csv", index=False)

#combine datasets
combined_data = pd.concat([data_1, data_2])

# Specify the columns to standardize and normalize
columns_to_standardize = [
    'age','sex','cp','trestbps','chol',
    'fbs','restecg','thalch','exang',
    'oldpeak','slope','ca','thal'
]
# columns_to_normalize = ["height", "weight"]

# Standardize the specified columns
scaler = StandardScaler()
combined_data[columns_to_standardize] = scaler.fit_transform(combined_data[columns_to_standardize])

# Normalize the specified columns
# normalizer = MinMaxScaler()
# df[columns_to_normalize] = normalizer.fit_transform(df[columns_to_normalize])

# Save the standardized and normalized dataset to a CSV file
combined_data = combined_data.astype(float)
combined_data = combined_data.astype({'num':'int64'})
combined_data.to_csv("standardized_data.csv", index=False)




