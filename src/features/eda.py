import matplotlib.pyplot as plt
import seaborn as sns

def eda_column_subset(df, list_of_columns):
    """
    Generates descriptive statistics, pair plot, and correlation matrix for a subset of columns.

    Accepts: dataframe, list of columns
    Returns: None, generates descriptive stats, pairplot, and correlation heat map
    """


    plt.figure(figsize=(40, 40))
    _, axs = plt.subplots()
    sns.heatmap(df[list_of_columns].corr(), annot=True, cmap=sns.color_palette("RdBu_r", 7), ax=axs)

    sns.pairplot(df[list_of_columns], plot_kws={'alpha': 0.2})
    return df[list_of_columns].describe([.1, .25, .5, .75, .9])
