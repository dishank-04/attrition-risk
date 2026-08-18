import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

def feature_skewness(df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:

    '''
    Calculates Skewness of Numeric Columns of Dataframe
    '''

    skewness = df[feature_cols].skew().sort_values(ascending=False) # Pandas has inbuilt skew() function to check for skewness, it only allows numeric cols

    return skewness



def plot_distribution(df: pd.DataFrame, feature_cols: list[str]) -> None:

    '''
    This will generate Histogram/KDE(Kernel Density Estimation) plots and Q-Q plots (for probability) side by side for each give feature. To check for Normal Distribution and Tails.
    '''

    n_features = len(feature_cols)
    fig, ax = plt.subplots(nrows=n_features, ncols=2, figsize=(12, 4*n_features), squeeze=False)

    for i, col in enumerate(feature_cols):

        hist_axis = ax[i,0]
        probplot_axis = ax[i,1]

        sns.histplot(data=df[col], bins=30, kde=True, ax=hist_axis, color='skyblue')
        hist_axis.set_title(f"{col} - Distribution (Skew: {df[col].skew():.2f})")

        stats.probplot(x=df[col], dist="norm", plot=probplot_axis)
        probplot_axis.set_title(f"{col} - Probability (Q-Q) Plot")


    plt.tight_layout()
    plt.show()



def check_multicollinearity(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:

    '''
    This function will 2 tests to check for multicollinearity among columns. 
    1) Spearman Correlation Heatmap
    2) Variance Inflation Factor
    '''

    X = df[feature_cols].copy()

    # Correlation Matrix 

    corr_matrix = X.corr(method="spearman")
    plt.figure(figsize=(8,6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Spearman Rank Correlation Matrix")
    plt.show()

    # VIF

    vif_data = pd.DataFrame()
    vif_data["Feature"] = feature_cols

    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(feature_cols))]

    vif_data.sort_values(by="VIF", ascending=False)

    return vif_data