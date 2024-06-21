import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np


def populate_label_columns(data, labels):
    for label in labels:
        data[f'Next years {label}'] = data.groupby('Player ID')[label].shift(-1)
    return data

# Custom Dataset class
class PlayerDataset(Dataset):
    def __init__(self, df, sequence_length, labels):
        self.dataframe = df
        self.sequence_length = sequence_length
        self.labels = []
        for label in labels:
            self.labels.append(f'Next years {label}')
        self.players = df['Player ID'].unique()

    def __len__(self):
        return len(self.players)

    def __getitem__(self, idx):
        player_data = self.dataframe[self.dataframe['Player ID'] == self.players[idx]].sort_values(by='Year')
        x = player_data.drop(columns=self.labels + ['Player ID', 'Name', 'Tm', 'Pos', 'No.', 'Awards', 'QBrec']).values
        y = player_data[self.labels].values
        
        x_seq, y_seq = [], []
        for i in range(len(x) - self.sequence_length):
            x_seq.append(x[i:i + self.sequence_length])
            y_seq.append(y[i:i + self.sequence_length])
        
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32)

def train_model(model, train_dataloader, criterion, optimizer, num_epochs=10):
    # model.train()
    # for epoch in range(num_epochs):
    #     for x_batch, y_batch in train_dataloader:
    #         x_batch, y_batch = x_batch.view(-1, x_batch.size(2), x_batch.size(3)), y_batch.view(-1, y_batch.size(2))
    #         optimizer.zero_grad()
    #         outputs = model(x_batch)
    #         loss = criterion(outputs, y_batch)
    #         loss.backward()
    #         optimizer.step()
    #     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    model.train()
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_dataloader:
            # Reshape x_batch to [batch_size, sequence_length, input_size]
            x_batch = x_batch.squeeze(0)  # Remove the extra dimension added by DataLoader
            y_batch = y_batch.squeeze(0)  # Remove the extra dimension added by DataLoader

            optimizer.zero_grad()
            outputs = model(x_batch)
            # Reshape y_batch to [batch_size, output_size]
            y_batch = y_batch.view(-1, y_batch.size(-1))
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # Only use the last output for prediction
        return out

def evaluate_model(model, test_dataloader, criterion):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for x_batch, y_batch in test_dataloader:
            x_batch, y_batch = x_batch.view(-1, x_batch.size(2), x_batch.size(3)), y_batch.view(-1, y_batch.size(2))
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
        print(f'Test Loss: {total_loss / len(test_dataloader):.4f}')

def main():
    position = 'QB'
    labels = ['Yds', 'TD']

    
    data = pd.read_csv(f'/Users/jakehirst/Desktop/fantasy_football_predictors/Yearly_statistics_clean/{position}_table_clean.csv', index_col=0)
    #make a column for next year's labels
    data = populate_label_columns(data, labels)
    
    # Split the data into training and test sets
    train_data = data[data['Year'] < 2021]
    test_data = data[data['Year'] == 2022]
    prediction_data = data[data['Year'] == 2023]


    sequence_length = 5  # Adjust this experimentally
    train_dataset = PlayerDataset(train_data, sequence_length, labels)
    test_dataset = PlayerDataset(test_data, sequence_length, labels)
    prediction_dataset = PlayerDataset(test_data, sequence_length, labels)
    
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)    
    prediction_dataloader = DataLoader(prediction_dataset, batch_size=1, shuffle=False)   
    
    
    input_size = train_dataloader.dataset[0][0].shape[-1]
    hidden_size = 64
    output_size = len(labels)
    num_layers = 1

    model = RNNModel(input_size, hidden_size, output_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_dataloader, criterion, optimizer, num_epochs=10)
    evaluate_model(model, test_dataloader, criterion)
    
    
    ''' Generate predictions for 2024 '''
    predictions_for_2024 = []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in prediction_dataloader:
            x_batch = x_batch.view(-1, x_batch.size(2), x_batch.size(3))
            outputs = model(x_batch)
            predictions_for_2024.append(outputs.cpu().numpy())
    
    # Convert predictions to a DataFrame for better visualization
    predictions_for_2024 = np.concatenate(predictions_for_2024, axis=0)
    predictions_df = pd.DataFrame(predictions_for_2024, columns=[f'Predicted Next Year {label}' for label in labels])
    predictions_df['Player ID'] = prediction_data['Player ID'].unique()

    print('here')

    
if __name__ == "__main__":
    main()