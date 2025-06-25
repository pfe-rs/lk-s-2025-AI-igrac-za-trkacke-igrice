import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ClassesML2 import *
from agent.device import get_device
from custom_arch import batched_forward

class CarGameAgent(nn.Module):
    def __init__(self, input_size):
        super(CarGameAgent, self).__init__()

        num_actions = 4  # Gas, Steer Left, Steer Right, Brake

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # Independent probabilities per action
        actions = torch.sigmoid(self.fc4(x))
        
        return actions

    def mutate(self, mutation_rate=0.05, mutation_strength=0.1):
        """
        Randomly mutates weights and biases of the model.

        Args:
            mutation_rate (float): Probability each weight/bias mutates
            mutation_strength (float): Standard deviation of noise added
        """
        for param in self.parameters():
            if len(param.shape) > 1:  # Weights
                mask = (torch.rand_like(param) < mutation_rate).float()
                noise = torch.randn_like(param) * mutation_strength
                param.data += mask * noise
            else:  # Biases
                mask = (torch.rand_like(param) < mutation_rate).float()
                noise = torch.randn_like(param) * mutation_strength
                param.data += mask * noise
    def run_in_environment(self, env, visualize=True, threshold=0.5, maxsteps=500, device="cuda"):
        # os.system('clear')
        # print(device)
        """
        Runs the model in the environment once until done.

        Args:
            env: Initialized environment
            visualize (bool): If True, shows pygame window
            threshold (float): Action activation threshold
            maxsteps (int): Max steps before auto-termination
            device (str): 'cpu' or 'cuda' for running the model
        """
        self.to(device)  # Move model to desired device

        if visualize:
            env.start_pygame()

        state = env.reset()
        total_reward = 0
        running = True

        while running:
            if visualize:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

            # Move state to device
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                action_probs = self(state_tensor)

            # Move actions back to CPU for compatibility with environment
            chosen_actions = (action_probs > threshold).int().squeeze(0).cpu().tolist()

            state, reward, done, steps, score = env.step(chosen_actions)
            total_reward += reward

            if visualize:
                env.render()

            if done or steps >= maxsteps:
                running = False
        self.to("cpu")
        env.close()
        return total_reward




class Population:
    def __init__(self, pop_size: int, input_size: int, mutation_rate=0.05, mutation_strength=0.1, elite_fraction=0.2):
        self.pop_size = pop_size
        self.input_size = input_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elite_fraction = elite_fraction

        self.models = [CarGameAgent(input_size) for _ in range(pop_size)]
        self.fitnesses = [0 for _ in range(pop_size)]

    def evaluate(self, env_fn,visualize,threshold, maxsteps: int, device: str):
        """Evaluate all models using provided environment generator."""
        for i, model in enumerate(self.models):
            env = env_fn()
            self.fitnesses[i] = model.run_in_environment(env,visualize,threshold,maxsteps,device)
            # (self, env, visualize=True, threshold=0.5,maxsteps=500)
    
    def evaluate_gpu(self, env_fn, threshold: float = 0.5, maxsteps: int = 500, device: str = get_device()):
        """
        Evaluate all models on GPU using true batching logic.

        Args:
            env_fn: function to create environments
            threshold: action activation threshold (float)
            maxsteps: max steps per episode
            device: "cuda" or "cpu"
        """
        all_envs = [env_fn() for _ in range(self.pop_size)]
        batch_size = self.pop_size

        # Move models to device once
        for model in self.models:
            model.to(device)

        self.fitnesses = [0.0 for _ in range(batch_size)]
        dones = [False for _ in range(batch_size)]

        # Reset all environments
        for env in all_envs:
            env.state = env.reset()

        for step in range(maxsteps):
            if all(dones):
                break  # All agents are done

            states = []
            for env in all_envs:
                states.append(torch.tensor(env.state, dtype=torch.float32))

            states_tensor = torch.stack(states).to(device)

            actions = batched_forward(states_tensor, self.models, device=device)

            for i, (env, action_probs) in enumerate(zip(all_envs, actions)):
                if not dones[i]:
                    chosen_actions = (action_probs > threshold).int().cpu().tolist()
                    state, reward, done, steps, score = env.step(chosen_actions)
                    env.state = state
                    self.fitnesses[i] += reward
                    dones[i] = done

        # Move models back to CPU
        # TODO: Review if actually need to send to cpu
        for model in self.models:
            model.to("cpu")

    # def next_generation(self):
    #     """Create next generation: elitism + mutation + crossover."""
    #     num_elite = max(1, int(self.pop_size * self.elite_fraction))
    #     sorted_indices = sorted(range(self.pop_size), key=lambda i: self.fitnesses[i], reverse=True)
    #     elite = [self.models[i] for i in sorted_indices[:num_elite]]

    #     next_gen = [self.clone_model(m) for m in elite]

    #     while len(next_gen) < self.pop_size:
    #         if random.random() < 0.5:
    #             # Crossover
    #             parent1 = random.choice(elite)
    #             child = 
    #         else:
    #             # Mutated clone
    #             parent = random.choice(elite)
    #             child = self.clone_model(parent)

    #         child.mutate(self.mutation_rate, self.mutation_strength)
    #         next_gen.append(child)

    #     self.models = next_gen
    #     self.fitnesses = [0 for _ in range(self.pop_size)]

    def next_generation(self):
        """Create next generation: elitism + cloning + mutation only (no crossover)."""
        num_elite = max(1, int(self.pop_size * self.elite_fraction))
        sorted_indices = sorted(range(self.pop_size), key=lambda i: self.fitnesses[i], reverse=True)
        elite = [self.models[i] for i in sorted_indices[:num_elite]]

        next_gen = [self.clone_model(m) for m in elite]  # Copy elite directly

        while len(next_gen) < self.pop_size:
            parent = random.choice(elite)
            child = self.clone_model(parent)
            child.mutate(self.mutation_rate, self.mutation_strength)
            next_gen.append(child)

        self.models = next_gen
        self.fitnesses = [0 for _ in range(self.pop_size)]


    def best_model(self):
        """Return best model and its fitness."""
        best_idx = max(range(self.pop_size), key=lambda i: self.fitnesses[i])
        return self.models[best_idx], self.fitnesses[best_idx]

    def clone_model(self, model):
        clone = CarGameAgent(self.input_size)
        clone.load_state_dict(model.state_dict())
        return clone
    
    def save_best_models(self, save_path: str, top_n: int = 5):
        """
        Saves the top N models based on fitness.

        Args:
            save_path (str): Folder path where models will be saved
            top_n (int): Number of best models to save
        """
        os.makedirs(save_path, exist_ok=True)

        sorted_indices = sorted(range(self.pop_size), key=lambda i: self.fitnesses[i], reverse=True)

        for rank, idx in enumerate(sorted_indices[:top_n]):
            model = self.models[idx]
            fitness = self.fitnesses[idx]
            filename = os.path.join(save_path, f"model_rank{rank+1}_fitness{fitness:.2f}.pth")
            torch.save(model.state_dict(), filename)

        print(f"Saved top {top_n} models to {save_path}")
