# Neural Network from Scratch (NumPy + Pandas)

This project implements a fully connected feedforward neural network trained on the MNIST handwritten digits dataset. It's built entirely from scratch using **NumPy** for computation and **Pandas** for data handling—no high-level machine learning libraries like TensorFlow or PyTorch are used.

The aim is to demonstrate deep learning fundamentals by manually coding each component: forward propagation, backpropagation, loss calculation, and weight updates using gradient descent.

---

## 🚀 Features

* Feedforward neural network with 1 hidden layer
* Fully vectorized implementation using NumPy
* Activation functions: ReLU, Softmax
* Loss function: Cross-Entropy
* Dataset: MNIST (CSV format)
* Accuracy tracking + loss visualization using Matplotlib
* Trained using batch gradient descent

---

## 🧠 Theory Summary

A feedforward neural network processes inputs through layers of neurons where each neuron applies a weighted sum and non-linear activation. During training:

* **Forward pass**: Calculates output probabilities from input features.
* **Loss function**: Measures how far predictions are from true labels (cross-entropy).
* **Backpropagation**: Uses chain rule to compute gradients of loss w.r.t. each weight.
* **Gradient descent**: Updates weights in the opposite direction of gradients to reduce loss.

This entire process is implemented explicitly using matrix operations.

---

## 📁 Project Structure

```
nn-from-scratch/
├── main.py           # Loads data, trains network, plots metrics
├── model.py          # Neural network class and logic
├── .gitignore        # Ignore virtual env and large files
├── README.md         # You’re here
└── mnist_train.csv   # Local file (ignored by Git)
```

> Note: `mnist_train.csv` is large and not included in this repo. You must download it manually (see below).

---

## 📦 Requirements

* Python 3.12+
* NumPy
* Pandas
* Matplotlib

### Install via pip:

```bash
pip install numpy pandas matplotlib
```

---

## 📥 Dataset Setup

1. Download `mnist_train.csv` and `mnist_test.csv` from [this repo](https://github.com/sharmaroshan/MNIST-Dataset) or any reliable MNIST CSV source.
2. Place both files in your project root directory.
3. Do **not** commit these files—they are large and listed in `.gitignore`.

---

## 🏃‍♂️ How to Run

```bash
python main.py
```

During training, you’ll see epoch-by-epoch logs with:

* Training loss
* Training accuracy
* Test accuracy

After training, two plots will appear:

* Loss vs Epochs
* Accuracy vs Epochs

---

## 📊 Example Output

```
Epoch 0: loss=2.305, train_acc=0.098, test_acc=0.101
Epoch 1: loss=1.940, train_acc=0.410, test_acc=0.423
...
Epoch 10: loss=0.470, train_acc=0.867, test_acc=0.862
```

*Final accuracy depends on the number of epochs, learning rate, and network size.*

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙋‍♀️ Acknowledgements

* [NumPy Documentation](https://numpy.org/doc/)
* [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
* GitHub Copilot (auto-completion support)

---

## 📌 TODO / Improvements

* Implement mini-batch training
* Add more hidden layers
* Add command-line args for model configuration
* Save model weights for reuse
