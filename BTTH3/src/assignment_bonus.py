import os
os.environ["MPLBACKEND"] = "Agg"
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/mpl"

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data_path = "data/data.json"
if not os.path.exists(data_path):
    data_path = os.path.join(os.path.dirname(__file__), "../data/data.json")

fig_dir = "figures"
if not os.path.exists(fig_dir):
    fig_dir = os.path.join(os.path.dirname(__file__), "../figures")
os.makedirs(fig_dir, exist_ok=True)

with open(data_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

X_list = []
y_list = []

music_types = {'CD', '1LP', '2LP', 'Cassette', '7in', 'Digital'}
clothing_types = {
    'T-Shirt', "Women's T-Shirt", 'Longsleeve T-Shirt', 
    'Hoodie (pullover)', 'Hoodie (zip up)', 'Crewneck', 
    'Sweatpants', 'Tank', "Women's Tank"
}

for item in raw_data:
    price_str = item.get("price", "N/A")
    if price_str == "N/A" or price_str is None:
        continue
    try:
        price = float(price_str)
    except ValueError:
        continue
        
    title = item.get("title", "")
    ptype = item.get("type", "")
    
    title_len = len(title)
    word_count = len(title.split())
    is_music = 1.0 if ptype in music_types else 0.0
    is_clothing = 1.0 if ptype in clothing_types else 0.0
    is_bundle = 1.0 if ("bundle" in title.lower() or "fan pack" in title.lower() or ptype == "Bundle") else 0.0
    
    X_list.append([title_len, word_count, is_music, is_clothing, is_bundle])
    y_list.append(price)

X = np.array(X_list, dtype=np.float64)
y = np.array(y_list, dtype=np.float64).reshape(-1, 1)

np.random.seed(42)
indices = np.arange(len(X))
np.random.shuffle(indices)

train_size = int(len(X) * 0.8)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

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
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=" * 10 + " BONUS ASSIGNMENT: CRAWLED DATA PRICE PREDICTION (100% NUMPY) " + "=" * 10)
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R2:   {r2:.4f}")

plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, color='royalblue', edgecolors='k', alpha=0.7, label='Dự đoán')
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, 'r--', label='Đường lý tưởng (y = x)')
plt.xlabel("Giá thực tế ($)")
plt.ylabel("Giá dự đoán ($)")
plt.title("So sánh Giá thực tế vs Giá dự đoán (Crawled Data Bonus)")
plt.legend()
plt.tight_layout()
bonus_plot_path = os.path.join(fig_dir, "bonus_regression.png")
plt.savefig(bonus_plot_path)
plt.close()

print(f"Plot saved to: {bonus_plot_path}")
