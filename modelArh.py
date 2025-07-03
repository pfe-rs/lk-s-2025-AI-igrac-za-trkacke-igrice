import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ClassesML2 import *
from agent.utils import get_device
from custom_arch import batched_forward
import matplotlib.pyplot as plt
import random

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

    def next_generation(self):
        """Create next generation: elitism + cloning + mutation only (no crossover)."""
        num_elite = max(1, int(self.pop_size * self.elite_fraction))
        sorted_indices = sorted(range(self.pop_size), key=lambda i: self.fitnesses[i], reverse=True)
        elite = [self.models[i] for i in sorted_indices[:num_elite]]

        next_gen = [self.clone_model(m) for m in elite]  # Copy elite directly

        # NOTE: Cross parents code part   
        # while len(next_gen) < self.pop_size:
        #     if random.random() < 0.5:
        #         # Crossover
        #         parent1 = random.choice(elite)
        #         child = 
        #     else:
        #         # Mutated clone
        #         parent = random.choice(elite)
        #         child = self.clone_model(parent)

        #     child.mutate(self.mutation_rate, self.mutation_strength)
        #     next_gen.append(child)

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

# def CarGameAgentDouble():
#     def __init__(self, input_size):
#         super(CarGameAgentDouble, self).__init__()

#         self.fc1 = nn.Linear(input_size, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, 64)

#         # Two separate heads:
#         self.fc_gas_brake = nn.Linear(64, 2)   # Outputs: Gas, Brake
#         self.fc_steering = nn.Linear(64, 2)    # Outputs: Steer Left, Steer Right

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
        
#         gas_brake = torch.sigmoid(self.fc_gas_brake(x))    # 2 outputs
#         steering = torch.sigmoid(self.fc_steering(x))      # 2 outputs
        
#         return gas_brake, steering
import torch
import torch.nn as nn
import torch.nn.functional as F

class CarGameAgentDouble(nn.Module):
    def __init__(self, input_size):
        super(CarGameAgentDouble, self).__init__()

        num_actions = 2  # [gas/brake] or [left/right]

        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 64)
        self.output = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        actions = self.output(x)
        return actions



class CombinedCarGameAgent(nn.Module):
    def __init__(self, model_gas_brake, model_left_right):
        super(CombinedCarGameAgent, self).__init__()
        
        self.model_gb = model_gas_brake  # Handles gas/brake
        self.model_lr = model_left_right  # Handles left/right

    def forward(self, x):
        gas_brake = torch.sigmoid(self.model_gb(x))  # Expected shape: [batch_size, 2]
        steering = torch.sigmoid(self.model_lr(x))  # Expected shape: [batch_size, 2]


        combined_output = torch.cat([gas_brake, steering], dim=1)  # Final shape: [batch_size, 4]
        
        return combined_output
    def run_in_environment(self, env, visualize=True, threshold=0.5, maxsteps=500, device="cuda"):
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

            os.system('clear')
            print(action_probs)
            print(chosen_actions)


            state, reward, done, steps, score = env.step(chosen_actions)
            total_reward += reward

            if visualize:
                env.render()

            if done or steps >= maxsteps:
                running = False
        self.to("cpu")
        env.close()
        return total_reward



class CarGameAgentDoubleMaybe(nn.Module):
    def __init__(self, input_size):
        super(CarGameAgentDoubleMaybe, self).__init__()

        num_actions = 3  # [gas, brake, neither]

        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 64)
        self.output = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        logits = self.output(x)  # Raw scores for each class
        return logits


class CarGameAgentDoubleMaybeSneaky(nn.Module):
    def __init__(self, input_size):
        super(CarGameAgentDoubleMaybeSneaky, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 3)

        self.lrelu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        x = self.lrelu(self.fc3(x))
        x = self.lrelu(self.fc5(x))
        x = self.lrelu(self.fc6(x))
        x = self.out(x)
        return x

class CarGameAgentDoubleMaybeSneakyBig(nn.Module):
    def __init__(self, input_size):
        super(CarGameAgentDoubleMaybeSneakyBig, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 3)

        self.lrelu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        x = self.lrelu(self.fc3(x))
        x = self.lrelu(self.fc4(x))

        x = self.out(x)
        return x



class CarGameAgentDoubleMaybeFrontMid(nn.Module):
    def __init__(self, input_size):
        super(CarGameAgentDoubleMaybeFrontMid, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 6)

        self.lrelu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        x = self.lrelu(self.fc3(x))
        x = self.out(x)
        return x

    def answer(self, x):
        return [x[0],x[1],x[3],x[4]]

    def run_in_environment(self, env, visualize=True, maxsteps=500, device="cuda",startvx=0,startstep=0,plotenzi_loc=None):
        """
        Runs the model in the environment once until done.

        Args:
            env: Initialized environment
            visualize (bool): If True, shows pygame window
            maxsteps (int): Max steps before auto-termination
            device (str): 'cpu' or 'cuda' for running the model
        """
        self.to(device)

        color_copy=env.car.color

        if visualize:
            env.start_pygame()

        env.init_replay()
        # state = env.get_replay()
        total_reward = 0
        running = True

        action_numbers=[[0,0,0],[0,0,0]]

        actions_save=[]

        


        while running:
            if visualize:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                env.car.color=color_copy

            

            # if env.car.vx<startvx and env.steps<startstep:
            #     force_it=True
            # else:
            #     force_it=False
            force_it=False
            state = env.get_replay()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            
            # os.system('clear')
            # print("------------------------------------------")
            # print(state)
            with torch.no_grad():
                probs = self(state_tensor)
            
            probs_gb = probs[0, :3]
            probs_lr = probs[0, 3:]

            # Apply softmax per group for interpretable probabilities
            probs_gb = torch.softmax(probs_gb, dim=0)
            probs_lr = torch.softmax(probs_lr, dim=0)

            # Choose actions based on argmax
            gb_choice = torch.argmax(probs_gb).item()  # 0=gas, 1=brake, 2=neither
            lr_choice = torch.argmax(probs_lr).item()  # 0=left, 1=right, 2=neither

            actions_save.append(probs_gb)

            chosen_actions = [False, False, False, False]  # [gas, brake, left, right]

            

            action_numbers[0][gb_choice]+=1
            action_numbers[1][lr_choice]+=1

            if gb_choice == 0:
                chosen_actions[0] = True
                
            elif gb_choice == 1:
                chosen_actions[1] = True
                if visualize:
                    env.car.color=([120,120,120])





            if lr_choice == 0:
                chosen_actions[2] = True
            elif lr_choice == 1:
                chosen_actions[3] = True


            
            

            



            # os.system('clear')
            if visualize:
                print(probs_gb*100)
                print(probs_lr*100)
                print(gb_choice)
                print(lr_choice)
                print(f"Gas/Brake probs: {probs_gb.cpu().numpy()}")
                print(f"Left/Right probs: {probs_lr.cpu().numpy()}")
                print(f"Chosen actions: {chosen_actions}")

            _, reward, done, steps, score = env.step(chosen_actions)
            env.update_replay()
            total_reward += reward

            if visualize:
                env.render()

            if done or steps >= maxsteps:
                running = False

        self.to("cpu")
        env.close()
        if visualize:
            print(action_numbers)
        if visualize and plotenzi_loc is not None:
            probs_gas = [p[0] for p in actions_save]
            probs_brake = [p[1] for p in actions_save]
            probs_neither = [p[2] for p in actions_save]

            # Gas Probability Plot
            plt.figure(figsize=(10, 4))
            plt.plot(probs_gas, color='green')
            plt.xlabel('Timestep')
            plt.ylabel('Gas Probability')
            plt.title('Gas Probability Over Time')
            plt.grid()
            plt.savefig(plotenzi_loc + "_gb_gas.png")
            plt.close()

            # Brake Probability Plot
            plt.figure(figsize=(10, 4))
            plt.plot(probs_brake, color='red')
            plt.xlabel('Timestep')
            plt.ylabel('Brake Probability')
            plt.title('Brake Probability Over Time')
            plt.grid()
            plt.savefig(plotenzi_loc + "_gb_brake.png")
            plt.close()

            # Neither gb Probability Plot
            plt.figure(figsize=(10, 4))
            plt.plot(probs_neither, color='blue')
            plt.xlabel('Timestep')
            plt.ylabel('Neither gb Probability')
            plt.title('Neither gb Probability Over Time')
            plt.grid()
            plt.savefig(plotenzi_loc + "_gb_neither.png")
            plt.close()




            # Left Probability Plot
            plt.figure(figsize=(10, 4))
            plt.plot(probs_gas, color='green')
            plt.xlabel('Timestep')
            plt.ylabel('Left Probability')
            plt.title('Left Probability Over Time')
            plt.grid()
            plt.savefig(plotenzi_loc + "_lr_left.png")
            plt.close()

            # Right Probability Plot
            plt.figure(figsize=(10, 4))
            plt.plot(probs_brake, color='red')
            plt.xlabel('Timestep')
            plt.ylabel('Right Probability')
            plt.title('Right Probability Over Time')
            plt.grid()
            plt.savefig(plotenzi_loc + "_lr_right.png")
            plt.close()

            # Neither lr Probability Plot
            plt.figure(figsize=(10, 4))
            plt.plot(probs_neither, color='blue')
            plt.xlabel('Timestep')
            plt.ylabel('Neither lr Probability')
            plt.title('Neither lr Probability Over Time')
            plt.grid()
            plt.savefig(plotenzi_loc + "_lr_neither.png")
            plt.close()

        
        return total_reward


class CarGameAgentDoubleMaybeSneakyMid(nn.Module):
    def __init__(self, input_size):
        super(CarGameAgentDoubleMaybeSneakyMid, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 3)

        self.lrelu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        x = self.lrelu(self.fc3(x))
        x = self.out(x)
        return x

class CarGameAgentDoubleMaybeSneakySmall(nn.Module):
    def __init__(self, input_size):
        super(CarGameAgentDoubleMaybeSneakySmall, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.out = nn.Linear(128, 3)

        self.lrelu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.lrelu(self.fc1(x))
        x = self.out(x)
        return x

class CombinedCarGameAgentMaybe(nn.Module):
    def __init__(self, model_gas_brake, model_left_right):
        super(CombinedCarGameAgentMaybe, self).__init__()
        
        self.model_gb = model_gas_brake  # Outputs logits for [gas, brake, neither]
        self.model_lr = model_left_right  # Outputs logits for [left, right, neither]

    def forward(self, x,force_it=False):
        logits_gb = self.model_gb(x)  # Shape: [batch_size, 3]
        logits_lr = self.model_lr(x)  # Shape: [batch_size, 3]

        probs_gb = F.softmax(logits_gb, dim=1)  # Convert to probabilities
        # probs_gb = torch.tensor([[0, 1, 0]], dtype=torch.float32).to(x.device)
        probs_lr = F.softmax(logits_lr, dim=1)

        # if force_it:
        #     probs_gb = torch.tensor([[1, 0, 0]], dtype=torch.float32).to(x.device)


        return probs_gb, probs_lr  # Return both probability distributions

    def run_in_environment(self, env, visualize=True, maxsteps=500, device="cuda",startvx=0,startstep=0,plotenzi_loc=None):
        """
        Runs the model in the environment once until done.

        Args:
            env: Initialized environment
            visualize (bool): If True, shows pygame window
            maxsteps (int): Max steps before auto-termination
            device (str): 'cpu' or 'cuda' for running the model
        """
        self.to(device)

        color_copy=env.car.color

        if visualize:
            env.start_pygame()

        state = env.reset()
        total_reward = 0
        running = True

        action_numbers=[[0,0,0],[0,0,0]]

        actions_save=[]

        while running:
            if visualize:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                env.car.color=color_copy

            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)


            # if env.car.vx<startvx and env.steps<startstep:
            #     force_it=True
            # else:
            #     force_it=False
            force_it=False

            with torch.no_grad():
                probs_gb, probs_lr = self(state_tensor,force_it=force_it)
                

            # Choose actions based on argmax
            gb_choice = torch.argmax(probs_gb, dim=1).item()  # 0=gas, 1=brake, 2=neither
            lr_choice = torch.argmax(probs_lr, dim=1).item()  # 0=left, 1=right, 2=neither

            actions_save.append(probs_gb[0])

            chosen_actions = [False, False, False, False]  # [gas, brake, left, right]


            action_numbers[0][gb_choice]+=1
            action_numbers[1][lr_choice]+=1

            if gb_choice == 0:
                chosen_actions[0] = True
                
            elif gb_choice == 1:
                chosen_actions[1] = True
                if visualize:
                    env.car.color=([120,120,120])

            if lr_choice == 0:
                chosen_actions[2] = True
            elif lr_choice == 1:
                chosen_actions[3] = True



            

            



            # os.system('clear')

            # print(probs_gb*100)
            # print(probs_lr*100)
            # print(gb_choice)
            # print(lr_choice)
            # print(f"Gas/Brake probs: {probs_gb.cpu().numpy()}")
            # print(f"Left/Right probs: {probs_lr.cpu().numpy()}")
            # print(f"Chosen actions: {chosen_actions}")

            state, reward, done, steps, score = env.step(chosen_actions)
            total_reward += reward

            if visualize:
                env.render()

            if done or steps >= maxsteps:
                running = False

        self.to("cpu")
        env.close()
        print(action_numbers)
        if visualize and not plotenzi_loc ==None:
            probs_gas = [p[0] for p in actions_save]
            probs_brake = [p[1] for p in actions_save]
            probs_neither = [p[2] for p in actions_save]

            # Plotting
            plt.figure(figsize=(12, 6))
            plt.plot(probs_gas, label='Gas Probability', color='green')
            plt.plot(probs_brake, label='Brake Probability', color='red')
            plt.plot(probs_neither, label='Neither Probability', color='blue')

            plt.xlabel('Timestep')
            plt.ylabel('Probability')
            plt.title('Probability Evolution Over Time')
            plt.legend()
            plt.grid()

            plt.savefig(plotenzi_loc+".png")



        
        return total_reward


class CombinedCarGameAgentMaybeReplay(nn.Module):
    def __init__(self, model_gas_brake, model_left_right):
        super(CombinedCarGameAgentMaybeReplay, self).__init__()
        
        self.model_gb = model_gas_brake  # Outputs logits for [gas, brake, neither]
        self.model_lr = model_left_right  # Outputs logits for [left, right, neither]

    def forward(self, x,force_it=False):
        logits_gb = self.model_gb(x)  # Shape: [batch_size, 3]
        logits_lr = self.model_lr(x)  # Shape: [batch_size, 3]

        probs_gb = F.softmax(logits_gb, dim=1)  # Convert to probabilities
        # probs_gb = torch.tensor([[0, 1, 0]], dtype=torch.float32).to(x.device)
        probs_lr = F.softmax(logits_lr, dim=1)

        # if force_it:
        #     probs_gb = torch.tensor([[1, 0, 0]], dtype=torch.float32).to(x.device)


        return probs_gb, probs_lr  # Return both probability distributions

    def run_in_environment(self, env, visualize=True, maxsteps=500, device="cuda",startvx=0,startstep=0,plotenzi_loc=None):
        """
        Runs the model in the environment once until done.

        Args:
            env: Initialized environment
            visualize (bool): If True, shows pygame window
            maxsteps (int): Max steps before auto-termination
            device (str): 'cpu' or 'cuda' for running the model
        """
        self.to(device)

        color_copy=env.car.color

        if visualize:
            env.start_pygame()

        env.init_replay()
        # state = env.get_replay()
        total_reward = 0
        running = True

        action_numbers=[[0,0,0],[0,0,0]]

        actions_save=[]

        


        while running:
            if visualize:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                env.car.color=color_copy

            

            # if env.car.vx<startvx and env.steps<startstep:
            #     force_it=True
            # else:
            #     force_it=False
            force_it=False
            state = env.get_replay()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            
            # os.system('clear')
            # print("------------------------------------------")
            # print(state)
            with torch.no_grad():
                probs_gb, probs_lr = self(state_tensor,force_it=force_it)
                

            # Choose actions based on argmax
            gb_choice = torch.argmax(probs_gb, dim=1).item()  # 0=gas, 1=brake, 2=neither
            lr_choice = torch.argmax(probs_lr, dim=1).item()  # 0=left, 1=right, 2=neither

            actions_save.append(probs_gb[0])

            chosen_actions = [False, False, False, False]  # [gas, brake, left, right]

            

            action_numbers[0][gb_choice]+=1
            action_numbers[1][lr_choice]+=1

            if gb_choice == 0:
                chosen_actions[0] = True
                
            elif gb_choice == 1:
                chosen_actions[1] = True
                if visualize:
                    env.car.color=([120,120,120])





            if lr_choice == 0:
                chosen_actions[2] = True
            elif lr_choice == 1:
                chosen_actions[3] = True


            
            

            



            # os.system('clear')
            if visualize:
                print(probs_gb*100)
                print(probs_lr*100)
                print(gb_choice)
                print(lr_choice)
                print(f"Gas/Brake probs: {probs_gb.cpu().numpy()}")
                print(f"Left/Right probs: {probs_lr.cpu().numpy()}")
                print(f"Chosen actions: {chosen_actions}")

            _, reward, done, steps, score = env.step(chosen_actions)
            env.update_replay()
            total_reward += reward

            if visualize:
                env.render()

            if done or steps >= maxsteps:
                running = False

        self.to("cpu")
        env.close()
        if visualize:
            print(action_numbers)
        if visualize and plotenzi_loc is not None:
            probs_gas = [p[0] for p in actions_save]
            probs_brake = [p[1] for p in actions_save]
            probs_neither = [p[2] for p in actions_save]

            # Gas Probability Plot
            plt.figure(figsize=(10, 4))
            plt.plot(probs_gas, color='green')
            plt.xlabel('Timestep')
            plt.ylabel('Gas Probability')
            plt.title('Gas Probability Over Time')
            plt.grid()
            plt.savefig(plotenzi_loc + "_gb_gas.png")
            plt.close()

            # Brake Probability Plot
            plt.figure(figsize=(10, 4))
            plt.plot(probs_brake, color='red')
            plt.xlabel('Timestep')
            plt.ylabel('Brake Probability')
            plt.title('Brake Probability Over Time')
            plt.grid()
            plt.savefig(plotenzi_loc + "_gb_brake.png")
            plt.close()

            # Neither gb Probability Plot
            plt.figure(figsize=(10, 4))
            plt.plot(probs_neither, color='blue')
            plt.xlabel('Timestep')
            plt.ylabel('Neither gb Probability')
            plt.title('Neither gb Probability Over Time')
            plt.grid()
            plt.savefig(plotenzi_loc + "_gb_neither.png")
            plt.close()




            # Left Probability Plot
            plt.figure(figsize=(10, 4))
            plt.plot(probs_gas, color='green')
            plt.xlabel('Timestep')
            plt.ylabel('Left Probability')
            plt.title('Left Probability Over Time')
            plt.grid()
            plt.savefig(plotenzi_loc + "_lr_left.png")
            plt.close()

            # Right Probability Plot
            plt.figure(figsize=(10, 4))
            plt.plot(probs_brake, color='red')
            plt.xlabel('Timestep')
            plt.ylabel('Right Probability')
            plt.title('Right Probability Over Time')
            plt.grid()
            plt.savefig(plotenzi_loc + "_lr_right.png")
            plt.close()

            # Neither lr Probability Plot
            plt.figure(figsize=(10, 4))
            plt.plot(probs_neither, color='blue')
            plt.xlabel('Timestep')
            plt.ylabel('Neither lr Probability')
            plt.title('Neither lr Probability Over Time')
            plt.grid()
            plt.savefig(plotenzi_loc + "_lr_neither.png")
            plt.close()

        
        return total_reward













    








