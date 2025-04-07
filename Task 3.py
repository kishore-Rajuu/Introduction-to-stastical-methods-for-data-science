import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

sns.set_theme(style="whitegrid")
sns.set_palette("rocket")

plot_hist = sns.histplot
plot_kde2d = sns.kdeplot
plot_qq = stats.probplot
plot_line = plt.plot
plot_scatter = sns.scatterplot
plot_box = sns.boxplot

features = pd.read_csv("../X.csv")
target = pd.read_csv("../y.csv")
dataset = pd.concat([features, target], axis=1)

sns.set_palette("rocket")
model_to_use = "ModelC"
true_theta = np.array([0.8, -1.4, 0.9, 0.5, 12.0])
num_draws = 10_000
target_accepts = 500

x1_vals = dataset['x1'].values
x2_vals = dataset['x2'].values
y_obs = dataset['y'].values
n_obs = len(y_obs)


def generate_features(model_name, x1, x2, n):
    intercept = np.ones(n)
    if model_name == "ModelA":
        return np.c_[x1 ** 3, x1 ** 5, x2, intercept]
    elif model_name == "ModelB":
        return np.c_[x1, x2, intercept]
    elif model_name == "ModelC":
        return np.c_[x1, x1 ** 2, x1 ** 4, x2, intercept]
    elif model_name == "ModelD":
        return np.c_[x1, x1 ** 2, x1 ** 3, x1 ** 5, x2, intercept]
    else:
        return np.c_[x1, x1 ** 3, x1 ** 4, x2, intercept]

def sample_prior(center_value, scale=0.2, num_samples=10_000):
    spread = scale * abs(center_value)
    return np.random.uniform(center_value - spread, center_value + spread, num_samples)


important_indices = np.argsort(np.abs(true_theta))[-2:][::-1]
base_design_matrix = generate_features(model_to_use, x1_vals, x2_vals, n_obs)
fixed_params = true_theta.copy()

theta1_prior_samples = sample_prior(true_theta[important_indices[0]], num_draws)
theta2_prior_samples = sample_prior(true_theta[important_indices[1]], num_draws)

accepted_theta1 = []
accepted_theta2 = []

for i in range(num_draws):
    test_params = fixed_params.copy()
    test_params[important_indices[0]] = theta1_prior_samples[i]
    test_params[important_indices[1]] = theta2_prior_samples[i]

    y_simulated = base_design_matrix @ test_params
    dist = np.sum((y_obs - y_simulated) ** 2)

    if dist < 1e5:
        accepted_theta1.append(theta1_prior_samples[i])
        accepted_theta2.append(theta2_prior_samples[i])
        if len(accepted_theta1) >= target_accepts:
            break

accepted_theta1 = np.array(accepted_theta1)
accepted_theta2 = np.array(accepted_theta2)

plot_hist(accepted_theta1, kde=True, color='tomato', edgecolor='black')
plot_hist(accepted_theta2, kde=True, color='darkorange', edgecolor='black')
plot_kde2d(x=accepted_theta1, y=accepted_theta2, fill=True, cmap="rocket")

print("\nTotal Accepted Samples:", len(accepted_theta1))
