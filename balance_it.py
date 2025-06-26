import os
import pickle
import random
import torch

os.system('clear')

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
n_outputs = 4  # forward, brake, left, right

# === Load All Data ===
all_states = []
all_actions = []
numbers = [0, 0, 0, 0]

for filename in os.listdir(recordings_folder):
    if filename.endswith(".pkl"):
        with open(os.path.join(recordings_folder, filename), "rb") as f:
            record = pickle.load(f)

            for state, action in record:
                all_states.append(state)
                all_actions.append([float(a) for a in action])
                numbers = [x + y for x, y in zip(numbers, action)]

print(f"Loaded {len(all_states)} total samples.")
print(f"Total action counts: {numbers}")

# === Keep All Rare Actions + 10% of Gas-Only Frames ===
balanced_states = []
balanced_actions = []
numbers = [0, 0, 0, 0]

for state, action in zip(all_states, all_actions):
    if action[1] or action[2] or action[3]:  # brake, left, or right pressed
        balanced_states.append(state)
        balanced_actions.append(action)
    else:
        if random.random() < 0.1:  # Keep 10% of pure "gas-only" frames
            balanced_states.append(state)
            balanced_actions.append(action)
for state, action in zip(balanced_states, balanced_actions):
    numbers = [x + y for x, y in zip(numbers, action)]

    

print(f"Final action counts after filtering: {numbers}")
print(f"Final dataset size: {len(balanced_states)}")
