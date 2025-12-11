import pandas as pd
import numpy as np

from model import NeuralNetwork


train_df = pd.read_csv("mnist_train.csv")
test_df  = pd.read_csv("mnist_test.csv")

X_train = train_df.iloc[:, 1:].values / 255.0   # shape (60000, 784), normalize to [0,1]
y_train = train_df.iloc[:, 0].astype(int).values  # labels 0–9
X_test  = test_df.iloc[:, 1:].values / 255.0    # shape (10000, 784)
y_test  = test_df.iloc[:, 0].astype(int).values

X_train = train_df.iloc[:,1:].to_numpy(dtype=np.float64) / 255.0
y_train = train_df.iloc[:,0].to_numpy(dtype=int)

num_classes = 10
Y_train = np.eye(num_classes)[y_train]   # shape (60000,10)
Y_test  = np.eye(num_classes)[y_test]

# Initialize network
nn = NeuralNetwork([784, 64, 10])        # 784 inputs, 64 hidden neurons, 10 outputs

epochs = 20
learning_rate = 0.01

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []


for epoch in range(epochs):
    # Forward and backward on training data
    Y_pred, caches = nn.forward(X_train)             # forward pass
    loss = nn.compute_loss(Y_train, Y_pred)          # compute loss
    grads_W, grads_b = nn.backprop(Y_train, caches)  # backpropagation
    nn.update_weights(grads_W, grads_b, learning_rate)
    
    # Compute training accuracy
    predictions = np.argmax(Y_pred, axis=1)
    accuracy = np.mean(predictions == y_train)
    
    # (Optionally evaluate on test set for monitoring)
    Y_test_pred, _ = nn.forward(X_test)
    test_acc = np.mean(np.argmax(Y_test_pred,axis=1) == y_test)

    train_losses.append(loss)

# Compute test loss for plotting
    test_loss = nn.compute_loss(Y_test, Y_test_pred)
    test_losses.append(test_loss)

    train_accuracies.append(accuracy)
    test_accuracies.append(test_acc)

    
    print(f"Epoch {epoch}: loss={loss:.3f}, train_acc={accuracy:.3f}, test_acc={test_acc:.3f}")

import matplotlib.pyplot as plt

epochs_range = range(1, epochs + 1)

# Loss plot
plt.figure(figsize=(8, 4))
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.plot(epochs_range, test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()
plt.grid(True)
plt.show()

# Accuracy plot
plt.figure(figsize=(8, 4))
plt.plot(epochs_range, train_accuracies, label="Train Accuracy")
plt.plot(epochs_range, test_accuracies, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()
plt.grid(True)
plt.show()
