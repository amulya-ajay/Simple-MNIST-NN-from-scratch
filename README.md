# 🧠 Simple MNIST Neural Network from Scratch

A simple **two-layer neural network built entirely from scratch using Python and NumPy** to recognize handwritten digits from the MNIST dataset.

The goal of this project is **not to build the most accurate MNIST classifier**, but to understand what happens inside a neural network — from input pixels to predictions, and from prediction errors back to updated weights.

Instead of using high-level deep learning frameworks such as PyTorch or TensorFlow, the core neural-network operations are implemented manually using matrix operations and NumPy.

> **The main objective:** understand the mathematics and mechanics behind neural networks rather than treating them as black boxes.

---

## 📌 Project Overview

The MNIST dataset contains grayscale images of handwritten digits ranging from **0 to 9**.

Each image is:

* **28 × 28 pixels**
* Flattened into **784 input features**
* Classified into one of **10 possible classes (0–9)**

This project implements the complete learning pipeline:

```text
MNIST Image
     │
     ▼
28 × 28 Image
     │
     ▼
Flatten → 784 Features
     │
     ▼
┌──────────────────────┐
│     Input Layer      │
│       784 neurons    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│     Hidden Layer     │
│       10 neurons     │
│        ReLU          │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│     Output Layer     │
│       10 neurons     │
│       Softmax        │
└──────────┬───────────┘
           │
           ▼
      Predicted Digit
       (0 – 9)
```

---

## ✨ What This Project Implements

The neural network is implemented without using a deep-learning framework.

### Core components

* Data loading and preprocessing
* Pixel normalization
* Train/development split
* Weight and bias initialization
* ReLU activation
* Softmax activation
* Forward propagation
* One-hot encoding
* Backpropagation
* Gradient descent
* Parameter updates
* Prediction
* Accuracy calculation
* Visualization of predictions

The implementation uses NumPy for the underlying matrix operations.

---

# 🏗️ Network Architecture

The network has the following architecture:

```text
784 → 10 → 10
```

### 1. Input Layer — 784 neurons

Every MNIST image has:

```text
28 × 28 = 784 pixels
```

The image is flattened into a vector:

```text
28 × 28 → 784
```

Each pixel becomes an input feature.

The input matrix therefore has the shape:

```text
784 × m
```

where `m` is the number of training examples.

---

### 2. Hidden Layer — 10 neurons

The first layer computes:

```text
Z₁ = W₁X + b₁
```

followed by the ReLU activation:

```text
A₁ = ReLU(Z₁)
```

The weight matrix has shape:

```text
W₁ : 10 × 784
```

and the bias has shape:

```text
b₁ : 10 × 1
```

---

### 3. Output Layer — 10 neurons

The second layer transforms the hidden representation:

```text
Z₂ = W₂A₁ + b₂
```

and applies Softmax:

```text
A₂ = Softmax(Z₂)
```

The ten output neurons correspond to:

```text
0 1 2 3 4 5 6 7 8 9
```

The class with the highest probability becomes the predicted digit.

---

# 🧮 How the Neural Network Learns

One of the main purposes of this project is to understand the learning process.

The network repeatedly performs three major steps:

```text
Forward Propagation
        ↓
Calculate Gradients
        ↓
Update Parameters
        ↓
Repeat
```

---

## 1. Forward Propagation

Given an input `X`, the network first calculates the hidden layer:

```text
Z₁ = W₁X + b₁
A₁ = ReLU(Z₁)
```

Then the output layer:

```text
Z₂ = W₂A₁ + b₂
A₂ = Softmax(Z₂)
```

`A₂` contains the predicted probabilities for each digit.

For example:

```text
[0.01, 0.02, 0.03, 0.80, 0.04, ...]
```

would indicate that the network believes the image is most likely:

```text
3
```

---

# 🔄 Backpropagation

Once a prediction has been generated, the network needs to determine:

> **How should each weight and bias change to make the prediction better?**

This is where **backpropagation** comes in.

The implementation calculates gradients by applying the chain rule backwards through the network.

For the output layer:

```text
dZ₂ = A₂ - Y
```

Then:

```text
dW₂ = (1/m) dZ₂ A₁ᵀ
```

and the gradient is propagated to the hidden layer:

```text
dZ₁ = W₂ᵀ dZ₂ ⊙ ReLU'(Z₁)
```

Finally:

```text
dW₁ = (1/m) dZ₁ Xᵀ
```

These gradients tell the network how its parameters contributed to the prediction error.

---

# 📉 Gradient Descent

After calculating the gradients, the parameters are updated using gradient descent:

```text
W₁ ← W₁ - α dW₁
b₁ ← b₁ - α db₁

W₂ ← W₂ - α dW₂
b₂ ← b₂ - α db₂
```

where:

* `α` = learning rate
* `dW` = gradient of the weights
* `db` = gradient of the biases

In the notebook, the network is trained with:

```text
Learning rate = 0.10
Iterations    = 500
```

The training accuracy improves progressively during these iterations.

---

# 📊 Results

The current implementation reaches approximately:

```text
Training Accuracy ≈ 85.1%
```

after 500 iterations on the training data.

The accuracy starts around random-guess territory and steadily improves:

```text
Iteration     Accuracy
-----------------------
0             ~8.5%
100           ~61.1%
200           ~75.4%
300           ~80.9%
400           ~83.6%
490           ~85.1%
```

These results demonstrate that the network is successfully learning useful patterns from the handwritten digits.

> **Note:** This is the training accuracy reported by the notebook. The current notebook does not provide a proper held-out test-set accuracy, so the ~85.1% figure should not be presented as the model's generalization accuracy.

---

# 🔍 Making Predictions

After training, the notebook provides a function for predicting individual images.

The process is:

```text
Input Image
     ↓
Forward Propagation
     ↓
Softmax Probabilities
     ↓
argmax()
     ↓
Predicted Digit
```

The notebook also visualizes the selected handwritten image alongside its prediction and actual label.

For example:

```text
Prediction: 4
Label:      4
```

This makes it possible to visually inspect what the network is predicting rather than looking only at an accuracy number.

---

# 🛠️ Technologies Used

| Technology       | Purpose                                           |
| ---------------- | ------------------------------------------------- |
| Python           | Programming language                              |
| NumPy            | Matrix operations and neural-network computations |
| Pandas           | Loading and manipulating the dataset              |
| Matplotlib       | Visualizing handwritten digits                    |
| Jupyter Notebook | Interactive development and experimentation       |

No high-level neural-network framework is required.

---

# 📁 Project Structure

```text
Simple-MNIST-NN-from-scratch/
│
├── Neural Network.ipynb
├── train.csv
└── README.md
```

### `Neural Network.ipynb`

Contains the complete implementation, including:

* Data preprocessing
* Neural-network architecture
* Activation functions
* Forward propagation
* Backpropagation
* Gradient descent
* Training
* Prediction
* Visualization

### `train.csv`

The handwritten-digit training dataset used by the notebook.

---

# 🚀 Getting Started

## 1. Clone the repository

```bash
git clone https://github.com/amulya-ajay/Simple-MNIST-NN-from-scratch.git
cd Simple-MNIST-NN-from-scratch
```

## 2. Install dependencies

```bash
pip install numpy pandas matplotlib jupyter
```

## 3. Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Neural Network.ipynb
```

Make sure `train.csv` is available in the expected location before running the notebook.

---

# 🎯 Learning Objectives

This project was built primarily as a learning exercise.

It helped explore the fundamentals behind:

### Mathematics

* Matrix multiplication
* Vectors and matrices
* Derivatives
* Chain rule
* Gradients
* Gradient descent

### Neural Networks

* Neurons
* Weights and biases
* Activation functions
* Forward propagation
* Backpropagation
* Parameter optimization

### Machine Learning

* Training data
* Features and labels
* Normalization
* Classification
* One-hot encoding
* Model evaluation

The purpose is to make the transition from:

```text
"Neural networks are something provided by a library"
```

to:

```text
"I understand what the library is actually doing."
```

---

# 🧠 Why Build a Neural Network From Scratch?

Modern frameworks make neural networks extremely easy to create.

For example, frameworks such as PyTorch can hide most of the implementation details behind a few lines of code.

That's useful for building real applications, but it can make it difficult to understand what is actually happening underneath.

This project takes the opposite approach.

Instead of:

```python
model.fit(X, y)
```

the learning process is explicitly implemented:

```text
Initialize parameters
        ↓
Forward propagation
        ↓
Calculate gradients
        ↓
Backpropagate
        ↓
Update parameters
        ↓
Repeat
```

This makes the mathematics behind neural-network training much more tangible.

---

# ⚠️ Current Limitations

This implementation is intentionally simple and has several limitations.

### 1. Very small hidden layer

The architecture is:

```text
784 → 10 → 10
```

A larger hidden layer could learn a richer representation of the input.

### 2. Full-batch gradient descent

The entire training set is processed during each iteration.

Modern implementations generally use mini-batches for better computational efficiency and optimization behavior.

### 3. No dedicated test evaluation

The current notebook evaluates accuracy primarily on the training data.

A proper ML workflow should evaluate the final model on unseen test data.

### 4. Simple parameter initialization

The weights are initialized using a simple random distribution.

More sophisticated initialization strategies such as Xavier/Glorot or He initialization can provide better training behavior.

### 5. Educational implementation

The implementation prioritizes **clarity and learning over production-level numerical stability and performance**.

For example, the Softmax implementation can be made more numerically stable by subtracting the maximum logit before exponentiation.

---

# 🚧 Possible Improvements

This project can be extended significantly.

Some natural next steps are:

* [ ] Add a proper validation/test set
* [ ] Implement a loss function explicitly
* [ ] Add loss tracking and visualization
* [ ] Implement mini-batch gradient descent
* [ ] Increase hidden-layer size
* [ ] Add multiple hidden layers
* [ ] Experiment with different learning rates
* [ ] Implement Xavier/He initialization
* [ ] Improve numerical stability of Softmax
* [ ] Add confusion matrix evaluation
* [ ] Compare different activation functions
* [ ] Implement momentum
* [ ] Implement Adam optimizer
* [ ] Compare the implementation against a PyTorch model
* [ ] Build a simple interface for drawing and classifying digits

These improvements would gradually transform this educational implementation into a more complete neural-network project.

---

# 📚 Concepts Covered

```text
                    Neural Networks
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   Architecture       Forward Pass       Training
        │                  │                  │
   Input Layer          Z = WX+b         Backpropagation
   Hidden Layer         ReLU             Gradients
   Output Layer         Softmax          Gradient Descent
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                     Classification
                           │
                         MNIST
```

---

# 🌱 What's Next?

This project is a foundation for understanding deeper machine-learning systems.

After understanding this implementation, the natural progression is:

```text
Neural Network From Scratch
            ↓
Better Neural Network Architecture
            ↓
Mini-Batch Training
            ↓
Optimization Algorithms
            ↓
Regularization
            ↓
Deep Neural Networks
            ↓
Convolutional Neural Networks
            ↓
Computer Vision
```

The same fundamental ideas explored here — **forward propagation, gradients, backpropagation, and optimization** — form the foundation of much more sophisticated deep-learning systems.

---

## 📌 Final Note

This repository is primarily a **learning project**.

The goal wasn't to achieve state-of-the-art MNIST accuracy. The goal was to remove the abstraction layer and understand how a neural network actually learns from data.

> **Don't just use neural networks. Understand them.**

---

## 👨‍💻 Author

**Amulya Ajay**

Computer Science Engineering Student
Interested in **Data Science, Machine Learning, Artificial Intelligence, and Software Development**.

---

⭐ If you found this project useful for learning neural networks, consider giving the repository a star.
