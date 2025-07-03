import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from modelArh import CarGameAgentDoubleMaybeFrontMid,CarGameAgentDoubleMaybeSneakyMid
from balance_it import *
from GATrainer import get_last_gen
from agent.const import *
import time









batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ray_number = 7
parametri = 6
stanja = 4
n_outputs = 4  # 4 actions: forward, brake, left, right
meower=parametri+stanja+2*ray_number
# n_inputs = parametri + stanja + 2 * ray_number
n_inputs=int(10+last_size)*(meower+n_outputs)+meower

# === Load and balance data using your function ===






    

# === Parameters ===
recordings_folder = "clean-codes/recordings_replay/"
epochs = 1000
learning_rate = 0.001
pos_weight = torch.tensor([0.02,5,1,5,5,0.02], device=device)  
# === Initialize Model ===
model = CarGameAgentDoubleMaybeFrontMid(n_inputs).to(device)
foldeer="models_supervised_absolute/"
output_file="output5.txt"
gen=1820






all_states, all_actions = load_dataset_clean(recordings_folder=recordings_folder)

print(f"Loaded {len(all_states)} samples after balancing.")

numbers_per=[0,0,0,0,0,0]

for action in  all_actions:
    for i in range(6):
        numbers_per[i]+=action[i]

print(numbers_per)


# === Convert to Tensors ===
states_tensor = torch.tensor(all_states, dtype=torch.float32)
actions_tensor = torch.tensor(all_actions, dtype=torch.float32)

# === Create Dataset and DataLoader ===
dataset = TensorDataset(states_tensor, actions_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)











# folder_path = "models_supervised_maybe_big/"



if not os.path.exists(foldeer):
    os.makedirs(foldeer)

model.load_state_dict(torch.load("models_supervised_absolute/model"+str(gen)+".pkl", map_location=device))



criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# === Training Loop ===
model.train()
for epoch in range(epochs):
    start_time = time.time()

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

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
    with open(output_file, "a") as f:
        # print(str(epoch)+": "+str(mean_value), file=f)
        print(f"Epoch {epoch+1+gen}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}", file = f)


    print(f"Time taken: {time.time() - start_time:.6f} seconds")

    filename=foldeer+"model"+str(epoch+gen)+".pkl"

    torch.save(model.state_dict(), filename)

print("Training complete.")

