import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

# Parameters
N_TROOP_TYPES = 16   # adjust to your total troop types
K_HAND_CARDS = 8     # adjust to total possible cards
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Brain(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 1 + 4 + N_TROOP_TYPES + 4*N_TROOP_TYPES + K_HAND_CARDS
        hidden_size = 64

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.card_head = nn.Linear(hidden_size, 5)   # 0-4 cards
        self.zone_head = nn.Linear(hidden_size, 4)   # 4 drop zones

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        card_logits = self.card_head(x)
        zone_logits = self.zone_head(x)
        card_probs = F.softmax(card_logits, dim=-1)
        zone_probs = F.softmax(zone_logits, dim=-1)
        return card_probs, zone_probs

# Load or initialize model
MODEL_PATH = "brain_weights.pth"
brain = Brain().to(DEVICE)
optimizer = optim.Adam(brain.parameters(), lr=1e-3)

if os.path.exists(MODEL_PATH):
    brain.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("Loaded existing brain weights")
else:
    print("Starting fresh brain")

def select_action(state_tensor):
    state_tensor = state_tensor.to(DEVICE)
    card_probs, zone_probs = brain(state_tensor)
    
    # create distributions
    card_dist = torch.distributions.Categorical(card_probs)
    zone_dist = torch.distributions.Categorical(zone_probs)

    # sample actions
    card_action = card_dist.sample()
    zone_action = zone_dist.sample()

    # log probability (needed later for RL)
    log_prob = card_dist.log_prob(card_action) + zone_dist.log_prob(zone_action)

    zone_action_for_play = zone_action.item() + 1

    return card_action.item(), zone_action_for_play, log_prob

def update_policy(log_probs, rewards, gamma=0.99):
    """
    Perform a policy gradient update on the brain using collected actions and rewards.
    
    log_probs: list of log_prob tensors from each action
    rewards: list of rewards corresponding to each action
    gamma: discount factor for future rewards (optional)
    """
    if not log_probs or not rewards:
        print("No actions/rewards collected. Skipping update.")
        return

    # Compute discounted rewards
    discounted_rewards = []
    R = 0
    for r in rewards[::-1]:  # start from last reward
        R = r + gamma * R
        discounted_rewards.insert(0, R)

    # Convert to tensor and normalize for stability
    discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(DEVICE)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

    # Compute policy loss
    loss = 0
    for log_prob, R in zip(log_probs, discounted_rewards):
        loss += -log_prob * R  # negative because optimizer minimizes

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save the updated weights
    torch.save(brain.state_dict(), MODEL_PATH)
    print("Policy updated and weights saved.")

    # Clear lists for next episode
    log_probs.clear()
    rewards.clear()
