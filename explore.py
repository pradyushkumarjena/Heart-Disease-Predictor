import pandas as pd
df = pd.read_csv("data/heart.csv")
print(df.head())
print(df.info())
print(df['target'].value_counts(normalize=True))
print(df.isnull().sum())
