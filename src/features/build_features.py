# Import libraries
import pandas as pd
import numpy as np

# Load the data
df = pd.read_pickle("../../data/interim/01_data_cleaned.pkl")

# Value counts for categorical column
df.CourseCategory.value_counts(normalize=True)

# One Hot Encoding for CourseCategory column
df = pd.get_dummies(df, columns=["CourseCategory"], dtype="int32")

# Export dataframe to pickle file
df.to_pickle("../../data/processed/02_data_processed.pkl")
