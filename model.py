import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# Device configuration: use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# 1. Load Data
df = pd.read_csv(r'data\g1_Patient_15_1.csv')
df['DeviceDtTm'] = pd.to_datetime(df['DeviceDtTm'])
df = df.sort_index()


# 2. Feature Extraction
glucose = df['Glucose'].values.reshape(-1, 1)


# 3. Scaling
scaler = MinMaxScaler()
glucose_scaled = scaler.fit_transform(glucose)


# 4. Create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)


SEQ_LEN = 10
X, y = create_sequences(glucose_scaled, SEQ_LEN)
X = torch.FloatTensor(X)
y = torch.FloatTensor(y)


# 5. Dataset & DataLoader
class GlucoseDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = GlucoseDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# 6. Model Definition
class GlucoseLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=1, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        out = self.fc(out)
        return out


# --- Define model path ---
model_save_path = 'glucose_lstm_model.pth'

# --- Initialize model ---
model = GlucoseLSTM().to(device)

# --- Load saved model if it exists ---
if os.path.exists(model_save_path):
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    print(f"Loaded saved model from {model_save_path}")
else:
    print("No saved model found, will train a new model")


# --- Only train if no saved model was loaded ---
if not os.path.exists(model_save_path):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 20
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Batch Loss: {loss.item():.5f}")

        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {epoch_loss / len(dataloader):.5f}")

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


# 8. Prediction (CUDA aware)
model.eval()
with torch.no_grad():
    last_seq = torch.FloatTensor(glucose_scaled[-SEQ_LEN:]).unsqueeze(0).to(device)
    next_pred_scaled = model(last_seq).cpu().numpy()
    next_pred = scaler.inverse_transform(next_pred_scaled)
    print('Predicted next glucose value:', next_pred.ravel()[0])


# 9. Plot true vs predicted
preds = []
model.eval()
with torch.no_grad():
    for i in range(len(X)):
        xi = X[i].unsqueeze(0).to(device)
        pred = model(xi).cpu().item()
        preds.append(pred)
preds_unscaled = scaler.inverse_transform(np.array(preds).reshape(-1,1))


plt.figure(figsize=(14, 6))
plt.plot(df['DeviceDtTm'][SEQ_LEN:], glucose[SEQ_LEN:], label="True Glucose")
plt.plot(df['DeviceDtTm'][SEQ_LEN:], preds_unscaled, label="LSTM Predicted")
plt.legend()
plt.title('LSTM: True vs Predicted Glucose')
plt.xlabel('Time')
plt.ylabel('Glucose')
plt.show()


# Function to predict on new input dataframe with variable length glucose readings
def predict_from_dataframe(model, df_input, scaler, seq_len=SEQ_LEN, device=device):
    # Ensure df_input has 'Glucose' column and sorted by time if 'DeviceDtTm' exists
    if 'DeviceDtTm' in df_input.columns:
        df_input['DeviceDtTm'] = pd.to_datetime(df_input['DeviceDtTm'])
        df_input = df_input.sort_values('DeviceDtTm')
    glucose_vals = df_input['Glucose'].values.reshape(-1,1)
    
    # Scale glucose values (use existing scaler)
    glucose_scaled = scaler.transform(glucose_vals)

    # Make sure input length is enough for at least one sequence
    if len(glucose_scaled) < seq_len:
        raise ValueError(f"Input data length must be at least {seq_len}")

    # Create sequences for prediction (using sliding window)
    X_new, _ = create_sequences(glucose_scaled, seq_len)
    X_new = torch.FloatTensor(X_new).to(device)

    model.eval()
    preds_scaled = []
    with torch.no_grad():
        for i in range(len(X_new)):
            xi = X_new[i].unsqueeze(0)  # batch of 1
            pred = model(xi).cpu().numpy()
            preds_scaled.append(pred)
    
    preds_scaled = np.vstack(preds_scaled)
    preds_unscaled = scaler.inverse_transform(preds_scaled)
    return preds_unscaled.flatten()


# Function to test model prediction on random samples from the dataframe
def test_model_on_random_sample(model, df, scaler, seq_len=SEQ_LEN, device=device, sample_size=1):
    """
    Select random sequences from df and compare model predictions vs actual glucose values.
    Prints results for each sample.

    Args:
        model: Trained LSTM model
        df: Original dataframe with 'Glucose' and 'DeviceDtTm'
        scaler: fitted MinMaxScaler used for glucose scaling
        seq_len: sequence length used in model
        device: torch device
        sample_size: how many random samples to test (default 1)
    """
    model.eval()
    glucose_vals = df['Glucose'].values.reshape(-1, 1)
    scaled_glucose = scaler.transform(glucose_vals)

    for _ in range(sample_size):
        start_idx = random.randint(0, len(df) - seq_len - 1)
        input_seq = scaled_glucose[start_idx:start_idx + seq_len]
        true_next = glucose_vals[start_idx + seq_len][0]
        input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_scaled = model(input_tensor).cpu().numpy()

        pred = scaler.inverse_transform(pred_scaled)[0][0]
        time_stamp = df['DeviceDtTm'].iloc[start_idx + seq_len]

        print(f"Sample Timestamp: {time_stamp}")
        print(f"True next glucose: {true_next:.2f}")
        print(f"Predicted next glucose: {pred:.2f}")
        print(f"Absolute error: {abs(true_next - pred):.2f}\n")
        print(f"Percentage error: {((abs(true_next - pred) / true_next) * 100):.2f}")


# Example usage of testing function (uncomment to run after training):
test_model_on_random_sample(model, df, scaler, sample_size=10)

# Example usage of prediction function on new data:
# new_df = pd.DataFrame({'Glucose': [value1, value2, value3, ...]})
# predicted_values = predict_from_dataframe(model, new_df, scaler)
# print(predicted_values)
