import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

data_path = "data/forestfires.csv"
if not os.path.exists(data_path):
    data_path = os.path.join(os.path.dirname(__file__), "../data/forestfires.csv")

df = pd.read_csv(data_path)

month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
day_map = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7}

df['month'] = df['month'].map(month_map)
df['day'] = df['day'].map(day_map)

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

X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

y_pred = X_test @ beta

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=" * 10 + " ASSIGNMENT 1: OLS MANUAL " + "=" * 10)
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")
