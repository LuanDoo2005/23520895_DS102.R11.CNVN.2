import os
import json
import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import LogisticsRegression

os.makedirs("figures", exist_ok=True)

def load_crawled_data(filepath="data/data.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    X_list = []
    y_list = []
    
    music_types = ["Digital", "1LP", "2LP", "CD", "Cassette", "7in"]
    
    for item in raw_data:
        price_str = item.get("price", "N/A")
        if price_str == "N/A":
            continue
            
        price = float(price_str)
        title_len = len(item.get("title", ""))
        ptype = item.get("type", "")
        
        # Binary label: 1 cho Am nhac, 0 cho Thoi trang & Khac
        y_label = 1 if ptype in music_types else 0
        
        X_list.append([price, title_len])
        y_list.append(y_label)
        
    X = np.array(X_list)
    y = np.array(y_list).reshape(-1, 1)
    
    # Standardization Z-score
    X_std = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-15)
    return X_std, y

X_bonus, y_bonus = load_crawled_data("data/data.json")

model_bonus = LogisticsRegression(epoch=200, lr=0.1)
model_bonus.fit(X_bonus, y_bonus)

pred_bonus = model_bonus.predict(X_bonus)
metrics = model_bonus.evaluate(y_bonus, pred_bonus)

print("=" * 10 + " BONUS ASSIGNMENT: CRAWLED DATA (100% NUMPY) " + "=" * 10)
for metric in metrics:
    print(f"{metric}:\t{metrics[metric]:.4f}")

plt.figure()
plt.plot(model_bonus.losses, color='green')
plt.title("Loss Function Curve - Crawled Data Bonus")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("figures/loss_bonus.png")
plt.close()
