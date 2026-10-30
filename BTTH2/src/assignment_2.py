import os
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from softmax_regression import SoftmaxRegression

os.makedirs("figures", exist_ok=True)

train_images = idx2numpy.convert_from_file("data/train-images-idx3-ubyte")
train_labels = idx2numpy.convert_from_file("data/train-labels-idx1-ubyte")

test_images = idx2numpy.convert_from_file("data/t10k-images-idx3-ubyte")
test_labels = idx2numpy.convert_from_file("data/t10k-labels-idx1-ubyte")

N, _, _ = train_images.shape
train_images = train_images.reshape(N, -1) / 255.0
N, _, _ = test_images.shape
test_images = test_images.reshape(N, -1) / 255.0

def one_hot_encoding(labels, num_classes=10):
    N = labels.shape[0]
    one_hot = np.zeros((N, num_classes))
    one_hot[np.arange(N), labels] = 1.0
    return one_hot

train_labels_onehot = one_hot_encoding(train_labels, 10)
test_labels_onehot = one_hot_encoding(test_labels, 10)

model = SoftmaxRegression(epoch=100, lr=0.1)
model.fit(train_images, train_labels_onehot)

predicted_labels = model.predict(test_images)
metrics = model.evaluate(test_labels_onehot, predicted_labels)

print("=" * 10 + " ASSIGNMENT 2 RESULTS " + "=" * 10)
for metric in metrics:
    print(f"{metric}:\t{metrics[metric]:.4f}")

predicted_classes = np.argmax(predicted_labels, axis=1)
cfm = confusion_matrix(test_labels, predicted_classes)
print("\nConfusion Matrix:")
print(cfm)

plt.figure(figsize=(8, 6))
sns.heatmap(cfm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Assignment 2")
plt.savefig("figures/cm_softmax.png")
plt.close()

plt.figure()
plt.plot(model.losses)
plt.title("Loss Function Curve - Softmax Regression")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("figures/loss_softmax.png")
plt.close()