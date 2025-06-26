import torch
from torch._prims_common import Tensor
import torch.nn.functional as F

def batched_linear(inputs, weights, biases):
    """
    Manual batched linear layer.

    Args:
        inputs: [B, in_features]
        weights: [B, out_features, in_features]
        biases: [B, out_features]

    Returns:
        outputs: [B, out_features]
    """
    return torch.bmm(weights, inputs.unsqueeze(2)).squeeze(2) + biases


def batched_forward(states: Tensor, models, device: str):
    """
    Runs a true batched forward pass with different models, same architecture.

    Args:
        states: [B, input_size] float tensor
        models: list of CarGameAgent instances
        device: "cuda" or "cpu"

    Returns:
        actions: [B, num_actions] float tensor
    """
    batch_size = len(models)
    input_size = states.shape[1]

    states = states.to(device)

    # Extract weights & biases from models
    w1 = torch.stack([m.fc1.weight.data for m in models]).to(device)
    b1 = torch.stack([m.fc1.bias.data for m in models]).to(device)

    w2 = torch.stack([m.fc2.weight.data for m in models]).to(device)
    b2 = torch.stack([m.fc2.bias.data for m in models]).to(device)

    w3 = torch.stack([m.fc3.weight.data for m in models]).to(device)
    b3 = torch.stack([m.fc3.bias.data for m in models]).to(device)

    w4 = torch.stack([m.fc4.weight.data for m in models]).to(device)
    b4 = torch.stack([m.fc4.bias.data for m in models]).to(device)

    # Batched Forward
    x = F.relu(batched_linear(states, w1, b1))
    x = F.relu(batched_linear(x, w2, b2))
    x = F.relu(batched_linear(x, w3, b3))
    actions = torch.sigmoid(batched_linear(x, w4, b4))

    return actions  # [B, num_actions]

