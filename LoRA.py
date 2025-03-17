import torch
import torch.nn as nn

# Simulate a pre-trained weight matrix (e.g., from a transformer layer)
d, k = 64, 32  # Input dim = 64, output dim = 32
W = torch.randn(d, k)  # Frozen pre-trained weights
W.requires_grad = False  # Freeze it

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha=1.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha
        
        # Frozen pre-trained weights
        self.W = nn.Parameter(torch.randn(in_dim, out_dim), requires_grad=False)
        
        # LoRA matrices A and B
        self.A = nn.Parameter(torch.randn(in_dim, rank))  # d x r
        self.B = nn.Parameter(torch.randn(rank, out_dim))  # r x k
    
    def forward(self, x):
        # Compute Wx + alpha * (A * B)x
        base_output = torch.matmul(x, self.W)  # Wx
        lora_adjustment = torch.matmul(x, self.A)  # x * A -> batch x r
        lora_adjustment = torch.matmul(lora_adjustment, self.B)  # (batch x r) * B -> batch x out_dim
        return base_output + self.alpha * lora_adjustment

# Instantiate the layer
rank = 4  # Small rank for efficiency
lora_layer = LoRALayer(d, k, rank, alpha=1.0)

# Generate synthetic data
n_samples = 1000
x = torch.randn(n_samples, d)  # Input data
W_true = torch.randn(d, k)  # "True" weights for the task
y = torch.matmul(x, W_true) + 0.1 * torch.randn(n_samples, k)  # Targets with noise

# Split into train/test
train_size = 800
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Optimizer (only A and B are trainable)
optimizer = torch.optim.Adam([lora_layer.A, lora_layer.B], lr=0.01)
criterion = nn.MSELoss()

# Training loop
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = lora_layer(x_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Test the model
with torch.no_grad():
    y_test_pred = lora_layer(x_test)
    test_loss = criterion(y_test_pred, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")