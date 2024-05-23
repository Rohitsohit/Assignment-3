import torch
from torch import nn
import numpy as np

def create_linear_regression_model(input_size, output_size):
    """
    Create a linear regression model with the given input and output sizes.
    Hint: use nn.Linear
    """
    model = nn.Linear(input_size, output_size)
    return model

def train_iteration(X, y, model, loss_fn, optimizer):
    # Compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def fit_regression_model(X, y, learning_rate=0.01, num_epochs=1000, print_interval=100):
    """
    Train the model for the given number of epochs.
    Hint: use the train_iteration function.
    Hint 2: while working you can use the print function to print the loss every 1000 epochs.
    Hint 3: you can use the previous_loss variable to stop the training when the loss is not changing much.
    """
    input_features = X.shape[1]
    output_features = y.shape[1]
    model = create_linear_regression_model(input_features, output_features)
    
    loss_fn = nn.MSELoss()  # Use mean squared error loss
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    previous_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        loss = train_iteration(X, y, model, loss_fn, optimizer)
        if abs(previous_loss - loss.item()) < 1e-5:  # Stop condition based on loss change
            print(f"Stopping training at epoch {epoch} because loss is not changing much.")
            break
        if epoch % print_interval == 0:  # Print loss every `print_interval` epochs
            print(f"Epoch {epoch}, Loss: {loss.item()}")
        previous_loss = loss.item()
    return model

# Prepare data
data = np.array([
    [24.0, 2.0, 1422.40],
    [24.0, 4.0, 1469.50],
    [16.0, 3.0, 1012.70],
    [25.0, 6.0, 1632.20],
    [16.0, 1.0, 952.20],
    [19.0, 2.0, 1117.70],
    [14.0, 3.0, 906.20],
    [22.0, 2.0, 1307.30],
    [25.0, 4.0, 1552.80],
    [12.0, 1.0, 686.70],
    [24.0, 7.0, 1543.40],
    [19.0, 1.0, 1086.50],
    [23.0, 7.0, 1495.20],
    [19.0, 5.0, 1260.70],
    [21.0, 3.0, 1288.10],
    [16.0, 6.0, 1111.50],
    [24.0, 5.0, 1523.10],
    [19.0, 7.0, 1297.40],
    [14.0, 4.0, 946.40],
    [20.0, 3.0, 1197.10]
])

# Separate features (X) and target (y)
X = torch.tensor(data[:, :2], dtype=torch.float32)  # First two columns
y = torch.tensor(data[:, 2:], dtype=torch.float32)  # Last column

# Train the model
model = fit_regression_model(X, y)

# Predict unknown values
def predict_vault_value(model, gold, silver):
    with torch.no_grad():
        input_tensor = torch.tensor([[gold, silver]], dtype=torch.float32)
        prediction = model(input_tensor)
        return prediction.item()

# Example usage:
unknown_vaults = [
    [20.0, 2.0],  # Example 1
    [22.0, 3.0],  # Example 2
    [18.0, 4.0]   # Example 3
]

for vault in unknown_vaults:
    gold, silver = vault
    value = predict_vault_value(model, gold, silver)
    print(f"Predicted value for vault with {gold} gold and {silver} silver coins: £{value:.2f}")
