import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from modelArh import CarGameAgentDouble  # Your custom model
from balance_it import load_and_balance_for_double
from GATrainer import get_last_gen

# === Parameters ===
recordings_folder = "clean-codes/recordings/"
batch_size = 64
epochs = 300
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gen = 3

ray_number = 7
parametri = 6
stanja = 4
n_inputs = parametri + stanja + 2 * ray_number

# === Load and unpack data ===
gas_brake, left_right = load_and_balance_for_double(recordings_folder)

gb_states = [s for s, _ in gas_brake]
gb_actions = [a for _, a in gas_brake]

lr_states = [s for s, _ in left_right]
lr_actions = [a for _, a in left_right]

print(f"Loaded {len(gb_states)} gas/brake samples after balancing.")
print(f"Loaded {len(lr_states)} left/right samples after balancing.")

# === Prepare datasets ===
gb_states_tensor = torch.tensor(gb_states, dtype=torch.float32)
gb_actions_tensor = torch.tensor(gb_actions, dtype=torch.float32)
gb_dataset = TensorDataset(gb_states_tensor, gb_actions_tensor)
gb_dataloader = DataLoader(gb_dataset, batch_size=batch_size, shuffle=True)

lr_states_tensor = torch.tensor(lr_states, dtype=torch.float32)
lr_actions_tensor = torch.tensor(lr_actions, dtype=torch.float32)
lr_dataset = TensorDataset(lr_states_tensor, lr_actions_tensor)
lr_dataloader = DataLoader(lr_dataset, batch_size=batch_size, shuffle=True)

# === Initialize models and optimizers ===
gb_model = CarGameAgentDouble(n_inputs).to(device)
lr_model = CarGameAgentDouble(n_inputs).to(device)

# models_supervised/gas_brake_model1.pkl models_supervised/steer_model1.pkl
gb_model.load_state_dict(torch.load("models_supervised/gas_brake_model"+str(gen)+".pkl", map_location=device))
lr_model.load_state_dict(torch.load("models_supervised/steer_model"+str(gen)+".pkl", map_location=device))

gb_criterion = nn.BCEWithLogitsLoss()
lr_criterion = nn.BCEWithLogitsLoss()

gb_optimizer = optim.Adam(gb_model.parameters(), lr=learning_rate)
lr_optimizer = optim.Adam(lr_model.parameters(), lr=learning_rate)

# === Training loop ===
for epoch in range(epochs):
    gb_epoch_loss = 0
    lr_epoch_loss = 0

    gb_model.train()
    lr_model.train()

    # # Train gas/brake model
    for batch_states, batch_actions in gb_dataloader:
        batch_states = batch_states.to(device)
        batch_actions = batch_actions.to(device)

        gb_optimizer.zero_grad()
        output = gb_model(batch_states)
        loss = gb_criterion(output, batch_actions)
        loss.backward()
        gb_optimizer.step()
        gb_epoch_loss += loss.item()

    # Train left/right model
    for batch_states, batch_actions in lr_dataloader:
        batch_states = batch_states.to(device)
        batch_actions = batch_actions.to(device)

        lr_optimizer.zero_grad()
        output = lr_model(batch_states)
        loss = lr_criterion(output, batch_actions)
        loss.backward()
        lr_optimizer.step()
        lr_epoch_loss += loss.item()

    gb_avg_loss = gb_epoch_loss / len(gb_dataloader)
    lr_avg_loss = lr_epoch_loss / len(lr_dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Gas/Brake Loss: {gb_avg_loss:.4f}, Left/Right Loss: {lr_avg_loss:.4f}")

# === Save models ===
os.makedirs("models_supervised/", exist_ok=True)


torch.save(gb_model.state_dict(), f"models_supervised/gas_brake_model{gen+1}.pkl")
torch.save(lr_model.state_dict(), f"models_supervised/steer_model{gen+1}.pkl")

print(f"Training complete. Models saved as gas_brake_model{gen+1}.pkl and steer_model{gen+1}.pkl")
