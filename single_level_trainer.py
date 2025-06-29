import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from modelArh import CarGameAgentDoubleMaybe,CombinedCarGameAgentMaybe  # Your custom model
from balance_it import load_and_balance_for_double,load_and_balance_for_double_maybe,load_only
from GATrainer import get_last_gen
from Functions import load_all_levels
from GATrainer import env_fn
import numpy as np
import time



# === Parameters ===
recordings_folder = "clean-codes/recordings3/"
batch_size = 64
epochs = 100
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gen = 100

ray_number = 7
parametri = 6
stanja = 4
n_inputs = parametri + stanja + 2 * ray_number

# # === Load and unpack data ===
# gas_brake, left_right = load_and_balance_for_double(recordings_folder)

# gb_states = [s for s, _ in gas_brake]
# gb_actions = [a for _, a in gas_brake]

# lr_states = [s for s, _ in left_right]
# lr_actions = [a for _, a in left_right]

# print(f"Loaded {len(gb_states)} gas/brake samples after balancing.")
# print(f"Loaded {len(lr_states)} left/right samples after balancing.")


# === Load and unpack data ===
import random

gas_brake, left_right = load_and_balance_for_double_maybe(recordings_folder)

gb_states = [s for s, _ in gas_brake]
lr_states = [s for s, _ in left_right]

gb_actions = []
lr_actions = []

# Counters for balancing
gb_counts = [0, 0, 0]  # Gas, Brake, Neither
lr_counts = [0, 0, 0]  # Left, Right, Neither

# Process Gas/Brake actions
for _, action in gas_brake:
    if action[0]:
        gb_actions.append(0)
        gb_counts[0] += 1
    elif action[1]:
        gb_actions.append(1)
        gb_counts[1] += 1
    else:
        gb_actions.append(2)
        gb_counts[2] += 1

# Process Left/Right actions
for _, action in left_right:
    if action[0]:
        lr_actions.append(0)
        lr_counts[0] += 1
    elif action[1]:
        lr_actions.append(1)
        lr_counts[1] += 1
    else:
        lr_actions.append(2)
        lr_counts[2] += 1

print(f"Initial GB Counts: {gb_counts}")
print(f"Initial LR Counts: {lr_counts}")
# print(gb_actions[:10])
# --- Boost "Neither" Class ---

def boost_class(states, actions, target_class, desired_total):
    combined = list(zip(states, actions))
    current_count = actions.count(target_class)

    while current_count < desired_total:
        for s, a in combined:
            if a == target_class:
                states.append(s)
                actions.append(a)
                current_count += 1
                if current_count >= desired_total:
                    break

# Desired balancing: max(all classes) count for each class
max_gb = max(gb_counts)
max_lr = max(lr_counts)




for i in range(3):
    boost_class(lr_states, lr_actions, i, max_lr)
    

# Optional: shuffle after boosting
# gb_states = [s for s, _ in gas_brake]
# gb_actions = [a for _, a in gas_brake]

combined_gb = list(zip(gb_states, gb_actions))
random.shuffle(combined_gb)
gb_states, gb_actions = zip(*combined_gb)


combined_lr = list(zip(lr_states, lr_actions))
random.shuffle(combined_lr)
lr_states, lr_actions = zip(*combined_lr)

# print(f"After Balancing - GB: {gb_actions.count(0)} Gas, {gb_actions.count(1)} Brake, {gb_actions.count(2)} Neither")
# print(f"After Balancing - LR: {lr_actions.count(0)} Left, {lr_actions.count(1)} Right, {lr_actions.count(2)} Neither")


print(f"Loaded {len(gb_states)} gas/brake samples after balancing.")
print(f"Loaded {len(lr_states)} left/right samples after balancing.")


bg_number=[0,0,0]

for action in gb_actions:
    bg_number[action]+=1

print(bg_number)

gb_actions=list(gb_actions)

print(gb_actions)


lr_number=[0,0,0]   

for action in lr_actions:
    lr_number[action]+=1

print(lr_number)

lr_actions=list(lr_actions)



# === Prepare datasets ===
gb_states_tensor = torch.tensor(gb_states, dtype=torch.float32)
gb_actions_tensor = torch.tensor(gb_actions, dtype=torch.long)
gb_dataset = TensorDataset(gb_states_tensor, gb_actions_tensor)
gb_dataloader = DataLoader(gb_dataset, batch_size=batch_size, shuffle=True)

lr_states_tensor = torch.tensor(lr_states, dtype=torch.float32)
lr_actions_tensor = torch.tensor(lr_actions, dtype=torch.long)
lr_dataset = TensorDataset(lr_states_tensor, lr_actions_tensor)
lr_dataloader = DataLoader(lr_dataset, batch_size=batch_size, shuffle=True)

# === Initialize models and optimizers ===
gb_model = CarGameAgentDoubleMaybe(n_inputs).to(device)
lr_model = CarGameAgentDoubleMaybe(n_inputs).to(device)

# models_supervised/gas_brake_model1.pkl models_supervised/steer_model1.pkl
# gb_model.load_state_dict(torch.load("models_supervised_maybe2/gas_brake_model"+str(gen)+".pkl", map_location=device))
lr_model.load_state_dict(torch.load("models_supervised_maybe2/steer_model"+str(44)+".pkl", map_location=device))

# Class order: [Gas (0), Brake (1), Neither (2)]
class_weights = torch.tensor([0.5, 4.2, 1.0], dtype=torch.float32).to(device)

gb_criterion = nn.CrossEntropyLoss()
lr_criterion = nn.CrossEntropyLoss()

gb_optimizer = optim.Adam(gb_model.parameters(), lr=learning_rate)
lr_optimizer = optim.Adam(lr_model.parameters(), lr=learning_rate)

levels=load_all_levels("clean-codes/levels1")


gb_epoch_loss=100
lr_epoch_loss=100

# === Training loop ===
for epoch in range(epochs):
    start_time = time.time()
    last_gb=gb_epoch_loss
    last_lr=lr_epoch_loss
    gb_epoch_loss = 0
    lr_epoch_loss = 0

    gb_model.train()
    lr_model.train()

    for batch_states, batch_actions in gb_dataloader:
        batch_states = batch_states.to(device)
        batch_actions = batch_actions.to(device)

        gb_optimizer.zero_grad()
        output = gb_model(batch_states)
        loss = gb_criterion(output, batch_actions)
        loss.backward()
        gb_optimizer.step()
        gb_epoch_loss += loss.item()


    # for batch_states, batch_actions in lr_dataloader:
    #     batch_states = batch_states.to(device)
    #     batch_actions = batch_actions.to(device)

    #     lr_optimizer.zero_grad()
    #     output = lr_model(batch_states)
    #     loss = lr_criterion(output, batch_actions)
    #     loss.backward()
    #     lr_optimizer.step()
    #     lr_epoch_loss += loss.item()

    gb_avg_loss = gb_epoch_loss / len(gb_dataloader)
    lr_avg_loss = lr_epoch_loss / len(lr_dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Gas/Brake Loss: {gb_avg_loss:.4f}, Left/Right Loss: {lr_avg_loss:.4f}")

    # if epoch%30==0:
    torch.save(gb_model.state_dict(), f"models_supervised_maybe3/gas_brake_model{gen+epoch}.pkl")
    # torch.save(lr_model.state_dict(), f"models_supervised_maybe2/steer_model{gen+epoch}.pkl")


    meow_scores=[]
    model = CombinedCarGameAgentMaybe(gb_model, lr_model)
    model.eval()

    for level in levels:
        meow_fit = model.run_in_environment(
        env_fn(), visualize=False, maxsteps=200, device=device
        )
        meow_scores.append(meow_fit)

    mean_value = np.mean(meow_scores)

    with open("output.txt", "a") as f:
        print(mean_value, file=f)

    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.6f} seconds")




# # === Save models ===
# os.makedirs("models_supervised/", exist_ok=True)


# torch.save(gb_model.state_dict(), f"models_supervised_maybe/gas_brake_model{999999}.pkl")
# torch.save(lr_model.state_dict(), f"models_supervised_maybe/steer_model{999999}.pkl")

# print(f"Training complete. Models saved as gas_brake_model{gen+1}.pkl and steer_model{gen+1}.pkl")



