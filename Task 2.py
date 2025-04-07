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

features = pd.read_csv("../X.csv")
target = pd.read_csv("../y.csv")
dataset = pd.concat([features, target], axis=1)

feat_1 = dataset['x1'].values
feat_2 = dataset['x2'].values
target_vals = dataset['y'].values
num_samples = len(target_vals)

def create_design_matrix(config_id):
    intercept = np.ones(num_samples)
    if config_id == 1:
        return np.column_stack((feat_1**3, feat_1**5, feat_2, intercept))
    elif config_id == 2:
        return np.column_stack((feat_1, feat_2, intercept))
    elif config_id == 3:
        return np.column_stack((feat_1, feat_1**2, feat_1**4, feat_2, intercept))
    elif config_id == 4:
        return np.column_stack((feat_1, feat_1**2, feat_1**3, feat_1**5, feat_2, intercept))
    elif config_id == 5:
        return np.column_stack((feat_1, feat_1**3, feat_1**4, feat_2, intercept))


def fit_model(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

def calc_rss(y_actual, y_est):
    residuals = y_actual - y_est
    rss_val = np.sum(residuals ** 2)
    return rss_val, residuals

def calc_log_likelihood(rss_val, n):
    var_est = rss_val / (n - 1)
    log_like = - (n / 2) * np.log(2 * np.pi) - (n / 2) * np.log(var_est) - rss_val / (2 * var_est)
    return log_like, var_est

def calc_aic_bic(log_like, num_params, n):
    aic_val = 2 * num_params - 2 * log_like
    bic_val = num_params * np.log(n) - 2 * log_like
    return aic_val, bic_val

# === Task 2.1: Parameter Estimation ===
param_results = {}
for config in range(1, 6):
    design = create_design_matrix(config)
    weights = fit_model(design, target_vals)
    param_results[config] = {'weights': weights}

print("Estimated Model Coefficients:")
pprint(param_results)

# === Task 2.2: RSS Calculation ===
rss_results = {}
for cfg, res in param_results.items():
    design = create_design_matrix(cfg)
    predictions = design @ res['weights']
    rss, residuals = calc_rss(target_vals, predictions)
    rss_results[cfg] = {
        'predicted': predictions,
        'rss': rss,
        'residuals': residuals
    }

print("RSS for Each Model:")
pprint(rss_results)

# === Task 2.3: Log Likelihood and Variance ===
log_like_results = {}
for cfg, res in rss_results.items():
    rss = res['rss']
    log_likelihood, variance = calc_log_likelihood(rss, num_samples)
    log_like_results[cfg] = {
        'log_likelihood': log_likelihood,
        'variance': variance
    }

print("Log-Likelihood and Variance:")
pprint(log_like_results)

# === Task 2.4: AIC/BIC ===
info_criteria = {}
for cfg in log_like_results:
    k = create_design_matrix(cfg).shape[1]
    ll = log_like_results[cfg]['log_likelihood']
    aic, bic = calc_aic_bic(ll, k, num_samples)
    info_criteria[cfg] = {'AIC': aic, 'BIC': bic}

print("AIC and BIC Results:")
pprint(info_criteria)

# === Task 2.5: Residual Diagnostic Plots ===
for cfg, res in rss_results.items():
    resid = res['residuals']
    plot_hist(resid, kde=True, color='tomato', edgecolor='black')
    plot_qq(resid, dist="norm", plot=plt)

# === Task 2.6: Summary of Models ===
summary_table = []
for cfg in range(1, 6):
    summary_table.append({
        'Model': cfg,
        'RSS': rss_results[cfg]['rss'],
        'Log-Likelihood': log_like_results[cfg]['log_likelihood'],
        'AIC': info_criteria[cfg]['AIC'],
        'BIC': info_criteria[cfg]['BIC']
    })

print("Model Comparison Summary:")
pprint(summary_table)

# === Task 2.7: Final Model Evaluation (Model 3) ===
X_model_final = create_design_matrix(3)
X_train, X_test, y_train, y_test = train_test_split(X_model_final, target_vals, test_size=0.3, random_state=42)

# Fit final model on training data
theta_final = fit_model(X_train, y_train)
y_test_pred = X_test @ theta_final

# Confidence Interval for Predictions
res_train = y_train - (X_train @ theta_final)
var_train = np.sum(res_train ** 2) / (len(y_train) - 1)
X_test_var = np.sum((X_test @ np.linalg.inv(X_train.T @ X_train)) * X_test, axis=1)
stderr_pred = np.sqrt(var_train * X_test_var)

# CI bounds
ci_low = y_test_pred - 1.96 * stderr_pred
ci_high = y_test_pred + 1.96 * stderr_pred

# Sort for plotting
sorted_idx = np.argsort(y_test_pred)
sorted_pred = y_test_pred[sorted_idx]
sorted_true = y_test[sorted_idx]
ci_low_sorted = ci_low[sorted_idx]
ci_high_sorted = ci_high[sorted_idx]

# === Plot Final Model Predictions with CI ===
plt.plot(sorted_pred, label='Predicted Signal', color='blue')
plt.fill_between(range(len(sorted_pred)), ci_low_sorted, ci_high_sorted, color='lightblue', alpha=0.5, label='95% CI')
plt.scatter(range(len(sorted_true)), sorted_true, color='red', s=15, label='Actual Signal')

