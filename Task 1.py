
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pprint import pprint
from sklearn.model_selection import train_test_split

sns.set_theme(style="whitegrid")
sns.set_palette("rocket")

plot_hist = sns.histplot
plot_kde2d = sns.kdeplot
plot_qq = stats.probplot
plot_line = plt.plot
plot_hist = sns.histplot
plot_scatter = sns.scatterplot
plot_box = sns.boxplot


timestamps = pd.read_csv("../time.csv")
features = pd.read_csv("../X.csv")
target = pd.read_csv("../y.csv")

combined_df = pd.concat([timestamps, features, target], axis=1)

plot_line(combined_df['time'], combined_df['x1'], linestyle='--', color='indigo', linewidth=2)

plot_line(combined_df['time'], combined_df['y'], linestyle='-', color='darkorange', linewidth=2)

plot_hist(combined_df['x1'], kde=True, edgecolor='black', color='teal')

plot_hist(combined_df['y'], kde=True, edgecolor='black', color='tomato')

plot_scatter(data=combined_df, x='x1', y='y', color='crimson', edgecolor='black', alpha=0.7)
corr_value = combined_df['x1'].corr(combined_df['y'])
print(f"Correlation between x1 and y: {corr_value:.3f}")


plot_box(data=combined_df, x='x2', y='y', palette='Set2')


for group_value, group_data in combined_df.groupby('x2'):
    color_scatter = 'purple' if group_value == 0 else 'darkgreen'
    color_hist = 'steelblue' if group_value == 0 else 'orangered'
    plot_scatter(data=group_data, x='x1', y='y', color=color_scatter, alpha=0.6)
    plot_hist(group_data['y'], kde=True, color=color_hist)

