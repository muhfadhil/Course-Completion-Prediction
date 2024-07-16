# Import libraries
import pandas as pd

# Load the data
df = pd.read_csv("../../data/raw/online_course_engagement_data.csv")

# Drop UserID column
df = df.drop(columns="UserID", axis=1)

# Common info the data
df.info()

# Check data duplicated
print(f"The number of data duplicated is {df.duplicated().sum()} data")

# Drop data duplicated
df = df.drop_duplicates().reset_index(drop=True)

# Check missing data for each column
df.isna().sum()

# Distribution of CourseCompletion column
df.CourseCompletion.value_counts(normalize=True)

# Export dataframe to csv file
df.to_pickle("../../data/interim/01_data_cleaned.pkl")
