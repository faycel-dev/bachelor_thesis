import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('/scratch/fharrathi/data/LibriSpeech/train-clean-100.trans.csv')

# Group the data by speaker IDs
grouped = df.groupby('speaker_id')

# Initialize empty lists to store training and validation data
train_data = []
val_data = []

# Iterate over each group (speaker)
for speaker_id, group_df in grouped:
    # Split the speaker's data into training and validation sets
    train_df, val_df = train_test_split(group_df, test_size=0.2, random_state=42)
    
    # Append the data to the respective lists
    train_data.append(train_df)
    val_data.append(val_df)

# Concatenate the data from all speakers into single DataFrames for training and validation sets
train_data = pd.concat(train_data)
val_data = pd.concat(val_data)
# Optionally, save the training and validation sets to separate CSV files
train_data.to_csv('/scratch/fharrathi/data/LibriSpeech/train_dataset1.csv', index=False)
val_data.to_csv('/scratch/fharrathi/data/LibriSpeech/val_dataset1.csv', index=False)
