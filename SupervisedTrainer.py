import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from modelArh import CarGameAgent

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


# === Load Recordings ===
all_states = []
all_actions = []

for filename in os.listdir(recordings_folder):
    if filename.endswith(".pkl"):
        with open(os.path.join(recordings_folder, filename), "rb") as f:
            record = pickle.load(f)
            for state, action in record:
                all_states.append(state)
                all_actions.append([float(a) for a in action])

print(f"Loaded {len(all_states)} samples from recordings.")




# === Convert to Tensors ===
states_tensor = torch.tensor(all_states, dtype=torch.float32)
actions_tensor = torch.tensor(all_actions, dtype=torch.float32)

# === Create Dataset and DataLoader ===
dataset = TensorDataset(states_tensor, actions_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === Initialize Model ===
model = CarGameAgent(n_inputs).to(device)
criterion = nn.BCEWithLogitsLoss()  # For multi-label binary output
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

print("Training complete.")
