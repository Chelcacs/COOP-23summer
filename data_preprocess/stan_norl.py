import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load CSV file
df = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")

# Specify the columns to standardize and normalize
columns_to_standardize = [
    "chest_pain_type", "resting_bp_s", "cholesterol", 
    "resting_ecg", "max_heart_rate", "oldpeak", "ST_slope"
]
# columns_to_normalize = ["height", "weight"]

# Standardize the specified columns
scaler = StandardScaler()
df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])

# Normalize the specified columns
# normalizer = MinMaxScaler()
# df[columns_to_normalize] = normalizer.fit_transform(df[columns_to_normalize])

# Save the standardized and normalized dataset to a CSV file
df.to_csv("standardized_data.csv", index=False)
