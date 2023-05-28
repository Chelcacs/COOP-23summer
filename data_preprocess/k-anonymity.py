import pandas as pd

def generalize(column, level):
    """
    A simple generalization function that replaces the last 'level' digits with asterisks
    """
    return column.astype(str).str[:-level] + '*'*level

def k_anonymize(df, k=2):
    """
    A function that achieves k-anonymity by generalizing columns
    """
    for column in df.columns:
        n_unique = df[column].nunique()
        if n_unique < k:
            # skip columns that already have less than k unique values
            continue
        level = 1
        while n_unique >= k:
            # generalize the column until there are k or fewer unique values
            df[column] = generalize(df[column], level)
            n_unique = df[column].nunique()
            level += 1
        break
    return df

# Load the dataset from a CSV file
df = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')

# Anonymize the dataset using k-anonymity
k_anonymized_df = k_anonymize(df, k=10)

# Save the anonymized dataset to a new CSV file
k_anonymized_df.to_csv('my_k_anonymized_dataset.csv', index=False)
