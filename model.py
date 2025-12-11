import numpy as np

# Activation functions
def sigmoid(x): 
    return 1 / (1 + np.exp(-x))
def dsigmoid(x):  # derivative
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)
def drelu(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Neural Network class
class NeuralNetwork:
    def __init__(self, layer_sizes):
        # layer_sizes = [input_dim, hidden_dim, output_dim]
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes)-1):
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        """Perform a forward pass. Returns output activations and caches for backprop."""
        a = X
        caches = {'A0': X}
        for i in range(len(self.weights)):
            W, b = self.weights[i], self.biases[i]
            Z = a.dot(W) + b
            caches[f'Z{i+1}'] = Z
            # Use ReLU for hidden layers, Softmax for output layer
            if i < len(self.weights)-1:
                a = relu(Z)
            else:
                a = softmax(Z)
            caches[f'A{i+1}'] = a
        return a, caches

    def compute_loss(self, Y_true, Y_pred):
        # Using categorical cross-entropy for classification
        m = Y_true.shape[0]
        log_likelihood = -np.log(Y_pred[range(m), Y_true.argmax(axis=1)])
        loss = np.sum(log_likelihood) / m
        return loss

    def backprop(self, Y_true, caches):
        """Compute gradients via backpropagation. Returns gradients for W and b."""
        grads_W = [None]*len(self.weights)
        grads_b = [None]*len(self.biases)
        m = Y_true.shape[0]
        
        # Output layer error (derivative of loss w.r.t. Z)
        A_out = caches[f'A{len(self.weights)}']  # softmax output
        dZ = A_out - Y_true                     # shape (m, output_dim)
        
        # Backprop through layers
        for i in reversed(range(len(self.weights))):
            A_prev = caches[f'A{i}']            # activation from previous layer (A0 = X)
            grads_W[i] = (A_prev.T.dot(dZ)) / m
            grads_b[i] = np.sum(dZ, axis=0, keepdims=True) / m
            if i > 0:
                Z_prev = caches[f'Z{i}']
                dZ = dZ.dot(self.weights[i].T) * drelu(Z_prev)  # propagate to previous layer
        return grads_W, grads_b

    def update_weights(self, grads_W, grads_b, lr):
        """Gradient descent step."""
        for i in range(len(self.weights)):
            self.weights[i] -= lr * grads_W[i]
            self.biases[i]  -= lr * grads_b[i]
