import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from modelArh import CarGameAgent
from balance_it import load_and_balance_recordings_new,load_and_balance_recordings_only_gas  # your function
from GATrainer import get_last_gen



# === Parameters ===
recordings_folder = "clean-codes/recordings/"
batch_size = 64
epochs = 100
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ray_number = 7
parametri = 6
stanja = 4
n_inputs = parametri + stanja + 2 * ray_number
n_outputs = 4  # 4 actions: forward, brake, left, right

# === Load and balance data using your function ===
all_states, all_actions = load_and_balance_recordings_only_gas(recordings_folder=recordings_folder)

print(f"Loaded {len(all_states)} samples after balancing.")

# === Convert to Tensors ===
states_tensor = torch.tensor(all_states, dtype=torch.float32)
actions_tensor = torch.tensor(all_actions, dtype=torch.float32)

# === Create Dataset and DataLoader ===
dataset = TensorDataset(states_tensor, actions_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === Initialize Model ===
model = CarGameAgent(n_inputs).to(device)

pos_weight = torch.tensor([0.5, 2.0, 2.0, 1.5], device=device)  

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pos_weight = torch.tensor([0.5, 2.0, 2.0, 1], device=device)  

# === Training Loop ===
model.train()
for epoch in range(epochs):
    epoch_loss = 0
    for batch_states, batch_actions in dataloader:
        batch_states = batch_states.to(device)
        batch_actions = batch_actions.to(device)

        optimizer.zero_grad()
        outputs = model(batch_states)
        loss = criterion(outputs, batch_actions)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

filename="models_supervised/"+"model"+str(get_last_gen("models_supervised/")+1)+".pkl"
torch.save(model.state_dict(), filename)

print("Training complete.")

