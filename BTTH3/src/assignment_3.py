import os
os.environ["MPLBACKEND"] = "Agg"
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/mpl"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

data_path = "data/forestfires.csv"
if not os.path.exists(data_path):
    data_path = os.path.join(os.path.dirname(__file__), "../data/forestfires.csv")

fig_dir = "figures"
if not os.path.exists(fig_dir):
    fig_dir = os.path.join(os.path.dirname(__file__), "../figures")
os.makedirs(fig_dir, exist_ok=True)

df = pd.read_csv(data_path)
month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
day_map = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7}
df['month'] = df['month'].map(month_map)
df['day'] = df['day'].map(day_map)

corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix - Forest Fires")
plt.tight_layout()
heatmap_path = os.path.join(fig_dir, "correlation_heatmap.png")
plt.savefig(heatmap_path)
plt.close()

df = df.drop(columns=['DC'])

X_full = df.drop(columns=['area']).values
y_full = df['area'].values.reshape(-1, 1)

np.random.seed(42)
indices = np.arange(len(X_full))
np.random.shuffle(indices)

train_size = int(len(X_full) * 0.8)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

X_train = X_full[train_idx]
y_train = y_full[train_idx]
X_test = X_full[test_idx]
y_test = y_full[test_idx]

mu_train = np.mean(X_train, axis=0)
sigma_train = np.std(X_train, axis=0)
sigma_train[sigma_train == 0] = 1e-15

X_train_std = (X_train - mu_train) / sigma_train
X_test_std = (X_test - mu_train) / sigma_train

X_train_std = np.hstack([np.ones((X_train_std.shape[0], 1)), X_train_std])
X_test_std = np.hstack([np.ones((X_test_std.shape[0], 1)), X_test_std])

beta = np.linalg.inv(X_train_std.T @ X_train_std) @ X_train_std.T @ y_train
y_pred = X_test_std @ beta

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=" * 10 + " ASSIGNMENT 3: FEATURE ENGINEERING (DROP DC) " + "=" * 10)
print(f"Heatmap saved to: {heatmap_path}")
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")
