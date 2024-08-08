import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split, TensorDataset
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from scipy import stats as st

class ForwardModelLearner(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 learning_rate=1e-7,
                 weight_decay=0.01,
                 hidden_dim=64, 
                 ):
        super(ForwardModelLearner, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.forward_model = self.generate_nn_architecture(input_dim, output_dim, hidden_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.96)
        self.loss_fn = nn.MSELoss()
        self.loss = None

    def generate_nn_architecture(self, input_dim, output_dim, hidden_dim):
        network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, output_dim)
        )
        return network.to(self.device)

    def forward(self, observation, action):
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if action.ndim == 1:
            action = action.unsqueeze(0)

        # Pass the vector through the network
        x = self.forward_model(torch.cat([observation, action], dim=1))
        return x.to(self.device)
    
class InverseModelLearner(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 learning_rate=1e-7,
                 weight_decay=0.01,
                 hidden_dim=64, 
                 ):
        super(InverseModelLearner, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.inverse_model = self.generate_nn_architecture(input_dim, output_dim, hidden_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.96)
        self.loss_fn = nn.MSELoss()
        self.loss = None

    def generate_nn_architecture(self, input_dim, output_dim, hidden_dim):
        network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(16, output_dim)
        )
        return network.to(self.device)

    def forward(self, observation, next_observation):
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if next_observation.ndim == 1:
            next_observation = next_observation.unsqueeze(0)

        # Pass the vector through the network
        x = self.inverse_model(torch.cat([observation, next_observation], dim=1))

        return x.to(self.device)

class DistributionModelLearner(nn.Module):
    def __init__(self, input_dim, output_dim, learning_rate=1e-3):
        super(DistributionModelLearner, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.96)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1).to(self.device)
        expected_error = self.network(x)
        return expected_error

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim, learning_rate=1e-3):
        super(FeatureExtractor, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.96)

    def forward(self, x):
        return self.network(x.to(self.device))

class ICM(nn.Module):
    def __init__(self, env, feature_dim, learning_rate=1e-4, eta=0.05):
        super(ICM, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        forward_input_dim = feature_dim + action_dim
        inverse_input_dim = feature_dim * 2
        self.batch_size = 64
        self.learning_rate = learning_rate

        self.feature_extractor = FeatureExtractor(state_dim, feature_dim, learning_rate)
        self.forward_model = ForwardModelLearner(forward_input_dim, feature_dim, learning_rate)
        self.inverse_model = InverseModelLearner(inverse_input_dim, action_dim, learning_rate)
        #self.error_model = DistributionModelLearner(forward_input_dim, 1, learning_rate)
        self.eta = eta  # Scaling factor for intrinsic reward

    def forward(self, state, action, next_state):
        # Step 1: Feature extraction
        state_features = self.feature_extractor(state)
        next_state_features = self.feature_extractor(next_state)

        # Step 2: Forward model prediction
        predicted_next_state_features = self.forward_model(state_features, action)

        # Step 3: Inverse model prediction
        predicted_action = self.inverse_model(state_features, next_state_features)

        # Step 4: Forward loss calculation
        forward_loss = self.forward_model.loss_fn(predicted_next_state_features, next_state_features)

        # Step 5: Inverse loss calculation
        inverse_loss = self.inverse_model.loss_fn(predicted_action, action)

        # Step 6: Intrinsic reward calculation
        intrinsic_reward = self.eta * forward_loss.detach()

        return forward_loss, inverse_loss, intrinsic_reward

    def train(self, replay_data, num_epochs=1):
        # Convert the observations, actions, and next observations to the appropriate device and data type
        if isinstance(replay_data.observations, dict):
            observations = replay_data.observations['observation'].float().to(self.device)
            next_observations = replay_data.next_observations['observation'].float().to(self.device)
        else:
            observations = replay_data.observations.float().to(self.device)
            next_observations = replay_data.next_observations.float().to(self.device)
        actions = replay_data.actions.float().to(self.device)

        # Initialize an optimizer for the ICM
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Start training for the specified number of epochs
        for epoch in range(num_epochs):
            epoch_forward_loss = 0.0
            epoch_inverse_loss = 0.0

            # Zero the gradients of the optimizer
            optimizer.zero_grad()

            # Forward pass through the model
            forward_loss, inverse_loss, _ = self.forward(observations, actions, next_observations)

            # Compute the average forward and inverse losses
            epoch_forward_loss += forward_loss.item()
            epoch_inverse_loss += inverse_loss.item()
            epoch_forward_loss /= len(observations)
            epoch_inverse_loss /= len(observations)

            # Update the forward and inverse losses of the model
            self.forward_loss = epoch_forward_loss
            self.inverse_loss = epoch_inverse_loss

            # Backpropagate the losses
            total_loss = forward_loss + inverse_loss
            total_loss.backward()

            # Update the parameters of the model
            optimizer.step()

            #print(f"Epoch {epoch} - Forward Loss: {epoch_forward_loss} - Inverse Loss: {epoch_inverse_loss}")


    def build_data_buffer(self, transitions):
        """
        Build a data buffer from the transitions list
        The input transitions list is a list of tuples of the form (observation, action, next_observation)
        The output data buffer is a tensor of size (num_transitions, observation_size + action_size + next_observation_size)
        """ 
        num_transitions = len(transitions)
        observation_size = transitions[0][0].shape[0]
        action_size = transitions[0][1].shape[0]
        next_observation_size = transitions[0][2].shape[0]

        data_buffer = torch.zeros((num_transitions, observation_size + action_size + next_observation_size))
        for i, (obs, act, next_obs) in enumerate(transitions):
            data_buffer[i, :observation_size] = torch.tensor(obs)
            data_buffer[i, observation_size:observation_size + action_size] = torch.tensor(act)
            data_buffer[i, observation_size + action_size:] = torch.tensor(next_obs)

        return data_buffer

    def generate_transitions(self, actions, observations):
        """
        Generate a list of transitions using the current state-action network (used in model-based RL)
        """
        # Actions and Observations sampled independently from the replay buffer
        actions = np.array(actions)
        observations = np.array(observations)
        pred_next_observations = np.array([])
        rewards = np.array([])
        dones = np.array([])
        # Generate the next observation, reward, and done flag for each action-observation pair
        for action, observation in zip(actions, observations):
            pred_next_obs = self.forward(observation, action)
            done = self.env.is_done(pred_next_obs)
            reward = self.env.compute_reward(pred_next_obs)
            # Append the next observation, reward, and done flag to the lists
            pred_next_observations = np.append(pred_next_observations, np.array(pred_next_obs))
            rewards = np.append(rewards, np.array(reward))
            dones = np.append(dones, np.array(done))
        data = (observations, actions, pred_next_observations, rewards, dones)
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def compute_prob_obs_distribution(self, old_obs, action, obs):
        # Model prediction
        old_obs = torch.from_numpy(old_obs.astype(np.float32))
        action = torch.from_numpy(action.astype(np.float32))
        obs = torch.from_numpy(obs.astype(np.float32))
        expected_obs = self.forward(old_obs, action)
        expected_mse = self.forward_mse(old_obs, action)
        # Averages the dict expected_mse
        # Expected_mse = sum(expected_mse.values())/len(expected_mse)
        # Computes a gaussian model of the probability for each sensor type in both dicts expected_obs and expected_mse
        is_anomaly = False
        variance = abs(expected_mse)
        gaussian = torch.distributions.Normal(expected_obs, variance)
        # Computes the treshold probability that the observed obs is an anomaly (i.e., within 2 standard deviations)
        treshold = gaussian.log_prob(expected_obs + 5*variance)
        treshold = torch.exp(treshold).mean()

        # Computes the probability of the observed obs for each sensor type
        prob = torch.exp(gaussian.log_prob(obs)).mean()
        if prob < treshold:
            #print("The observed obs is an anomaly for sensor type {}: ".format(key), prob < treshold)
            is_anomaly = True
        return is_anomaly

    def compute_prob_obs(self, old_obs, action, obs, threshold=0.01):
        # Model prediction
        old_obs = torch.from_numpy(old_obs.astype(np.float32))
        action = torch.from_numpy(action.astype(np.float32))
        obs = torch.from_numpy(obs.astype(np.float32))
        expected_obs = self.forward(old_obs, action)
        expected_mse = self.forward_mse(old_obs, action)

        is_anomaly= False
        variance = abs(expected_mse)
        # Computes the z-score relative to a gaussian distribution
        z_score = np.linalg.norm(obs.detach().numpy() - expected_obs[0].detach().numpy()) / np.sqrt(variance)
        # Prints whether the observed obs is an anomaly or not for each sensor type
        # Computes the probability of the observed obs for each sensor type given the error
        prob = st.norm.cdf(z_score) if z_score < 0 else 1 - st.norm.cdf(z_score)
        if prob < threshold*1/np.linalg.norm(action):
            is_anomaly = True
        return is_anomaly

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)

    def compute_mse(self, data):
        total_mse = 0.0
        num_batches = 0
        with torch.no_grad():
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i+self.batch_size]
                for obs, act, next_obs in batch:
                    obs, act, next_obs = obs.to(self.device), act.to(self.device), next_obs.to(self.device)
                    pred_next_obs = self.forward(obs, act)
                    mse = ((pred_next_obs - next_obs) ** 2).sum().item()
                    total_mse += mse
                    num_batches += 1
        return total_mse / num_batches

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))