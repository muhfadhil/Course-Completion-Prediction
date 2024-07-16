# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")

# Load the data
df = pd.read_pickle("../../data/interim/01_data_cleaned.pkl")

# Correlation
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(numeric_only=True), annot=True, vmin=-1, vmax=1)
plt.savefig("../../reports/figures/correlation.png", bbox_inches="tight", dpi=400)
plt.show()

# Univariate
# DeviceType and CourseCompletion include categorical columns
# Numeric columns
num_cols = [
    "TimeSpentOnCourse",
    "NumberOfVideosWatched",
    "NumberOfQuizzesTaken",
    "QuizScores",
    "CompletionRate",
]

# Distribution of each numeric columns
for col in num_cols:
    # Statistic descriptive
    average = df[col].mean()
    median = df[col].median()
    mode = df[col].mode()[0]
    std = df[col].std()

    # Make subplot for histogram
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(data=df, x=col, kde=True)
    plt.axvline(average, color="r", linestyle="solid", linewidth=3, label="Mean")
    plt.axvline(median, color="y", linestyle="dotted", linewidth=3, label="Median")
    plt.axvline(mode, color="b", linestyle="dashed", linewidth=3, label="Mode")
    plt.legend(
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.17),
        fancybox=True,
        shadow=True,
    )
    plt.savefig(f"../../reports/figures/hist_{col}.png", bbox_inches="tight", dpi=400)
    plt.show()

# Violin plot for numeric columns
plt.figure(figsize=(10, 7))
sns.violinplot(data=df, orient="y")
plt.savefig("../../reports/figures/violin.png", bbox_inches="tight", dpi=400)
plt.show()

# Categorical columns
cat_cols = ["CourseCategory", "DeviceType", "CourseCompletion"]

# Barchart plot for categorical columns
for col in cat_cols:
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.countplot(data=df, x=col)
    plt.axhline()
    plt.ylabel("number of users")
    plt.savefig(f"../../reports/figures/bar_{col}", bbox_inches="tight", dpi=400)
    plt.show()

# Bivariate
# Number of CourseCategory and DeviceType by CouseCompletion
for col in ["CourseCategory", "DeviceType"]:
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.countplot(data=df, x=col, hue="CourseCompletion")
    plt.legend(
        loc="upper center",
        ncol=2,
        bbox_to_anchor=[0.5, 1.17],
        fancybox=True,
        shadow=True,
    )
    plt.savefig(
        f"../../reports/figures/bar_{col}_CourseCompletion",
        bbox_inches="tight",
        dpi=400,
    )
    plt.show()

# Relation of 2 numeric column
# numeric columns and CompletionRate column
for col in num_cols:
    if col == "CompletionRate":
        continue

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(data=df, x=col, y="CompletionRate", hue="CourseCompletion")
    plt.legend(
        loc="upper center",
        ncol=2,
        bbox_to_anchor=[0.5, 1.15],
        fancybox=True,
        shadow=True,
    )
    plt.savefig(
        f"../../reports/figures/scatter_{col}_CompletionRate",
        bbox_inches="tight",
        dpi=400,
    )
    plt.show()

# ["NumberOfVideosWatched", "NumberOfQuizzesTaken"] and "TimeSpentOnCourse"
for col in ["NumberOfVideosWatched", "NumberOfQuizzesTaken"]:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(data=df, x=col, y="TimeSpentOnCourse", hue="CourseCompletion")
    plt.legend(
        loc="upper center",
        ncol=2,
        bbox_to_anchor=[0.5, 1.15],
        fancybox=True,
        shadow=True,
    )
    plt.savefig(
        f"../../reports/figures/scatter_{col}_TimeSpentOnCourse",
        bbox_inches="tight",
        dpi=400,
    )
    plt.show()

# Numeric columns by categorical columns
for cat_col in cat_cols:
    if cat_col == "CourseCompletion":
        continue

    for num_col in num_cols:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=df, x=cat_col, y=num_col, hue="CourseCompletion")
        plt.legend(
            loc="upper center",
            ncol=2,
            bbox_to_anchor=[0.5, 1.15],
            fancybox=True,
            shadow=True,
        )
        plt.savefig(
            f"../../reports/figures/bar_{num_col}_{cat_col}",
            bbox_inches="tight",
            dpi=400,
        )
        plt.show()
