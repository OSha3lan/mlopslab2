import pandas as pd
from sklearn.model_selection import train_test_split

# Load raw CSV data
raw_data_path = 'WineQT.csv'  
df = pd.read_csv(raw_data_path)


df = df.dropna()


# Split data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)