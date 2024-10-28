# Gradient Descent Lab: Visualizing the Learning Process

In this lab, we'll implement gradient descent for linear regression step by step. We'll visualize how the algorithm learns from data and converges to optimal parameters.

## Learning Objectives
- Implement gradient descent from scratch
- Understand how parameters evolve during training
- Visualize the relationship between model predictions and actual data
- Track the loss function's behavior over time

## Prerequisites
- Basic Python programming
- Understanding of linear algebra fundamentals
- Familiarity with NumPy

## Required Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
```

## Part 1: Data Generation

First, let's generate synthetic data following a linear relationship with some added noise:

```python
# Set random seed for reproducibility
np.random.seed(42)

# True parameters we want to learn
true_b = 0.5  # intercept
true_w = -3   # slope
N = 100       # number of data points

# Generate input features
x = np.random.rand(N, 1)

# Generate target values with some noise
epsilon = 0.1 * np.random.randn(N, 1)
y = true_b + true_w * x + epsilon

# Split into training and validation sets
idx = np.arange(N)
np.random.shuffle(idx)

# Use 80% for training
train_idx = idx[:int(N*.8)]
val_idx = idx[int(N*.8):]

x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]
```

## Part 2: Visualization Setup

Let's create functions to visualize our data and model's predictions:

```python
def plot_data_and_model(x, y, w, b, title=""):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, c='blue', label='Data points')
    
    # Generate points for the line
    x_line = np.array([0, 1])
    y_line = w * x_line + b
    plt.plot(x_line, y_line, 'r-', label=f'Model: y = {b:.2f} + {w:.2f}x')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_loss_history(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.title('Loss History')
    plt.grid(True)
    plt.show()
```

## Part 3: Gradient Descent Implementation

Now, let's implement gradient descent step by step:

### Step 0: Random Initialization
```python
np.random.seed(42)
b = np.random.randn(1)[0]  # Initialize intercept
w = np.random.randn(1)[0]  # Initialize slope

print(f"Initial parameters - b: {b:.4f}, w: {w:.4f}")
plot_data_and_model(x_train, y_train, w, b, "Initial Model")
```

### Step 1-5: Training Loop
```python
# Hyperparameters
learning_rate = 0.1
n_iterations = 1000

# Lists to store history
losses = []
params_history = []

for epoch in range(n_iterations):
    # Step 1: Forward pass (make predictions)
    y_pred = w * x_train + b
    
    # Step 2: Compute loss (MSE)
    error = y_pred - y_train
    loss = np.mean(error ** 2)
    losses.append(loss)
    
    # Step 3: Compute gradients
    b_grad = 2 * np.mean(error)
    w_grad = 2 * np.mean(x_train * error)
    
    # Step 4: Update parameters
    b = b - learning_rate * b_grad
    w = w - learning_rate * w_grad
    
    # Store parameters
    params_history.append((b, w))
    
    # Print progress every 100 iterations
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.6f}, b = {b:.4f}, w = {w:.4f}")
        plot_data_and_model(x_train, y_train, w, b, f"Model at Epoch {epoch}")

# Plot final model
plot_data_and_model(x_train, y_train, w, b, "Final Model")
plot_loss_history(losses)
```

## Part 4: Analysis Questions

After completing the implementation, answer the following questions:

1. How close are your learned parameters (b, w) to the true parameters (true_b, true_w)?
2. Why doesn't the model achieve exactly the true parameters?
3. What happens if you:
   - Increase/decrease the learning rate?
   - Add more noise to the data?
   - Use fewer/more training points?
4. Look at the loss history plot. Why does the loss decrease quickly at first and then more slowly?

## Part 5: Experiments

Try modifying these aspects of the implementation and observe the results:

1. Change the learning rate to 0.01 and 1.0
```python
learning_rates = [0.01, 0.1, 1.0]
# Run training loop with each learning rate
```

2. Modify the noise level in the data generation
```python
noise_levels = [0.01, 0.1, 0.5]
# Generate data with different noise levels
```

3. Use different random initializations
```python
np.random.seed(0)  # Try different seeds
```

## Extension Challenges

1. Implement mini-batch gradient descent instead of full-batch
2. Add momentum to the parameter updates
3. Implement early stopping using the validation set
4. Add regularization to prevent overfitting

## Conclusion

Through this lab, you've implemented gradient descent from scratch and visualized how it learns from data. You've seen how the algorithm:
- Starts from random parameters
- Iteratively improves its predictions
- Converges to parameters close to the true values

The visualization tools help build intuition about how gradient descent works and how different factors affect the learning process.

Remember that while this implementation uses a simple linear regression model, the same principles apply to more complex models like neural networks!
