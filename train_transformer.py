import torch
import torch.nn as nn
import torch.optim as optim

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_dim, num_transformer_blocks, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation='relu'
            ) for _ in range(num_transformer_blocks)
        ])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(embed_dim, 20)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.embedding(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)

# Parameters
input_dim = num_features  # Number of input features
embed_dim = 64
num_heads = 4
ff_dim = 128
num_transformer_blocks = 2
dropout = 0.1

# Model, loss function, and optimizer
model = TimeSeriesTransformer(input_dim, embed_dim, num_heads, ff_dim, num_transformer_blocks, dropout)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example training loop
num_epochs = 10
batch_size = 32

# Assuming X_train and y_train are your training data and targets
# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(X_train_tensor), batch_size):
        X_batch = X_train_tensor[i:i + batch_size]
        y_batch = y_train_tensor[i:i + batch_size]
        
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
