import os
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from logistic_regression import LogisticsRegression

os.makedirs("figures", exist_ok=True)

train_images = idx2numpy.convert_from_file("data/train-images-idx3-ubyte")
train_labels = idx2numpy.convert_from_file("data/train-labels-idx1-ubyte")
train_data = (train_images, train_labels)

test_images = idx2numpy.convert_from_file("data/t10k-images-idx3-ubyte")
test_labels = idx2numpy.convert_from_file("data/t10k-labels-idx1-ubyte")
test_data = (test_images, test_labels)

def filter_data(data, condition):
    images, labels = data
    new_images = images[labels == condition]
    new_labels = labels[labels == condition]
    return new_images, new_labels

train_images_0, train_labels_0 = filter_data(train_data, 0)
train_images_1, train_labels_1 = filter_data(train_data, 1)
train_images = np.concatenate([train_images_0, train_images_1], axis=0)
train_labels = np.concatenate([train_labels_0, train_labels_1], axis=0)

test_images_0, test_labels_0 = filter_data(test_data, 0)
test_images_1, test_labels_1 = filter_data(test_data, 1)
test_images = np.concatenate([test_images_0, test_images_1], axis=0)
test_labels = np.concatenate([test_labels_0, test_labels_1], axis=0)

N, _, _ = train_images.shape
train_images = train_images.reshape(N, -1) / 255.0
N, _, _ = test_images.shape
test_images = test_images.reshape(N, -1) / 255.0

train_labels = train_labels.reshape(-1, 1)
test_labels = test_labels.reshape(-1, 1)

model = LogisticsRegression(epoch=100, lr=0.1)
model.fit(train_images, train_labels)

predicted_labels = model.predict(test_images)
metrics = model.evaluate(test_labels, predicted_labels)

print("=" * 10 + " ASSIGNMENT 1 RESULTS " + "=" * 10)
for metric in metrics:
    print(f"{metric}:\t{metrics[metric]:.4f}")

cfm = confusion_matrix(test_labels, (predicted_labels >= 0.5).astype(int))
print("\nConfusion Matrix:")
print(cfm)

plt.figure(figsize=(5, 5))
sns.heatmap(cfm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Assignment 1")
plt.savefig("figures/cm_logistic.png")
plt.close()

plt.figure()
plt.plot(model.losses)
plt.title("Loss Function Curve - Logistic Regression")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("figures/loss_logistic.png")
plt.close()