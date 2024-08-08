import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split, TensorDataset
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from scipy import stats as st

class ForwardModelLearner(nn.Module):
    def __init__(self, 
                 observation_space, 
                 action_space, 
                 learning_rate=1e-7,
                 weight_decay=0.01,
                 hidden_dim=256, 
                 batch_size=64,
                 save_dir=None, 
                 normalize=False, 
                 ):
        super(ForwardModelLearner, self).__init__()
        self.env = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.normalize = normalize
        self.observation_space = observation_space
        self.action_space = action_space
        self.batch_size = batch_size
        self.save_dir = save_dir
        if observation_space is not None:
            self.observation_size = observation_space.shape[0]
            self.action_size = action_space.shape[0]
            if normalize:
                self.obs_mean = torch.Tensor(observation_space.low + observation_space.high).to(self.device) / 2.0
                self.obs_std = torch.Tensor(observation_space.high - observation_space.low).to(self.device) / 2.0
                self.act_mean = torch.Tensor(action_space.low + action_space.high).to(self.device) / 2.0
                self.act_std = torch.Tensor(action_space.high - action_space.low).to(self.device) / 2.0
            self.forward_model = self.generate_nn_architecture(hidden_dim=hidden_dim, regrouped_outputs=False)
            self.error_model = self.generate_nn_architecture(hidden_dim=64, regrouped_outputs=True)
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
            self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.96)
            self.loss_fn = nn.MSELoss()
            self.loss = None
            self.error_loss = None
            self.l2_reg = nn.Linear(hidden_dim, 1)

    def generate_nn_architecture(self, hidden_dim=256, regrouped_outputs=False):
        """
        Generate the neural network architecture to model each sensor type
        params:
            hidden_dim: the number of hidden units in each hidden layer
            regrouped_outputs: whether to regroup the outputs of the network into a single tensor
        """
        output_dim = 1 if regrouped_outputs else self.observation_space.shape[0]
        network = nn.Sequential(
            nn.Linear(self.observation_space.shape[0] + self.action_space.shape[0], hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, output_dim)
        )
        return network.to(self.device)

    def normalize_obs(self, obs):
        obs = (obs - self.obs_mean) / self.obs_std
        return obs

    def normalize_act(self, act):
        if isinstance(self.action_space, gym.spaces.MultiDiscrete):
            act = gym.spaces.flatten(self.action_space, act)
            act = torch.Tensor([((a - m) / s) for a, m, s in zip(act, self.act_mean, self.act_std)])
        else:
            act = (act - self.act_mean) / self.act_std
        return act

    def denormalize_obs(self, obs_norm):
        obs = (obs_norm * self.obs_std) + self.obs_mean
        return obs

    def denormalize_act(self, act_norm):
        if isinstance(self.action_space, gym.spaces.MultiDiscrete):
            act_norm = act_norm.tolist()
            act = gym.spaces.unflatten(self.action_space, act_norm)
            act = [a * s + m for a, m, s in zip(act, self.act_mean, self.act_std)]
            act = torch.Tensor(act)
        else:
            act = (act_norm * self.act_std) + self.act_mean
        return act

    def forward(self, observation, action):
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if action.ndim == 1:
            action = action.unsqueeze(0)

        # Pass the vector through the network
        x = self.forward_model(torch.cat([observation, action], dim=1))

        return x.to(self.device)
    
    def forward_mse(self, observation, action):
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if action.ndim == 1:
            action = action.unsqueeze(0)

        # Pass the vector through the network
        x = self.error_model(torch.cat([observation, action], dim=1))

        return x.to(self.device)

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

    def train(self, replay_data, num_epochs=1, validation=False, train_error_model=False):
        """
        Train the SAC agent on a dataset buffer
        """
        # Convert the observations, actions, and next observations to the appropriate device and data type
        observations = replay_data.observations.float().to(self.device)
        actions = replay_data.actions.float().to(self.device)
        next_observations = replay_data.next_observations.float().to(self.device)

        # Normalize the observations, actions, and next observations if normalization is enabled
        if self.normalize:
            observations = self.normalize_obs(observations)
            actions = self.normalize_act(actions)
            next_observations = self.normalize_obs(next_observations)

        # Initialize an optimizer for the error network if error computation is enabled
        if train_error_model:
            error_optimizer = optim.Adam(self.error_model.parameters(), lr=1e-5, weight_decay=self.weight_decay)

        # Initialize variables for validation if validation is enabled
        if validation:
            best_mse = float('inf')
            patience_counter = 0
            data = TensorDataset(observations, actions, next_observations)
            validation_data = random_split(data, [int(0.8*len(data)), int(0.2*len(data))])

        # Start training for the specified number of epochs
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_error_loss = 0.0 if train_error_model else None

            # Zero the gradients of the optimizer
            self.optimizer.zero_grad()

            # Zero the gradients of the error optimizer if error computation is enabled
            if train_error_model:
                error_optimizer.zero_grad()

            # Forward pass through the model
            pred_next_obs = self.forward(observations, actions)

            # Compute the error if error computation is enabled
            pred_next_error = self.forward_mse(observations, actions) if train_error_model else None

            # Compute the loss
            sensor_loss = self.loss_fn(pred_next_obs, next_observations)
            epoch_loss += sensor_loss.item()

            # Compute the error loss if error computation is enabled
            if train_error_model:
                sensor_error_output = torch.mean(pred_next_error)
                sensor_error_loss = self.loss_fn(sensor_error_output, self.loss_fn(pred_next_obs, next_observations))
                epoch_error_loss += sensor_error_loss.item()

            # Compute the average loss and error loss
            epoch_loss /= len(observations)
            epoch_error_loss /= len(observations) if train_error_model else None

            # Update the loss and error loss of the model
            self.loss = epoch_loss
            self.error_loss = epoch_error_loss if train_error_model else None

            # Backpropagate the loss
            sensor_loss.backward(retain_graph=True)

            # Backpropagate the error loss if error computation is enabled
            if train_error_model:
                sensor_error_loss.backward(retain_graph=True)

            # Update the parameters of the model
            self.optimizer.step()

            # Update the parameters of the error network if error computation is enabled
            if train_error_model:
                error_optimizer.step()

            # Update the learning rate
            self.scheduler.step()

            # Validate the model on the validation set if validation is enabled
            if validation:
                with torch.no_grad():
                    val_loss = 0.0
                    val_error_loss = 0.0 if train_error_model else None

                    # Compute the validation loss and error loss
                    for obs, act, next_obs in validation_data[1]:
                        obs, act, next_obs = obs.to(self.device), act.to(self.device), next_obs.to(self.device)
                        pred_next_obs = self.forward(obs, act)
                        pred_next_error = self.forward_mse(obs, act) if train_error_model else None
                        sensor_loss = self.loss_fn(pred_next_obs, next_obs)
                        val_loss += sensor_loss.item()
                        if train_error_model:
                            sensor_error_output = torch.mean(pred_next_error)
                            sensor_error_loss = self.error_loss_fn(sensor_error_output, self.compute_mse(pred_next_obs, next_obs))
                            val_error_loss += sensor_error_loss.item()

                    # Compute the average validation loss and error loss
                    val_loss /= len(validation_data[1])
                    val_error_loss /= len(validation_data[1]) if train_error_model else None

                    # Update the best MSE and reset the patience counter if the validation loss is less than the best MSE
                    if val_loss < best_mse:
                        best_mse = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    # Stop training if the patience counter exceeds a certain limit
                    if patience_counter >= 10:
                        print("Early stopping")
                        break

                    print(f"Epoch {epoch} - Validation Loss: {val_loss} - Validation Error Loss: {val_error_loss}")

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

    def save(self, path=None):
        """
        Save the model as a zip file
        """
        if path is None:
            path = f"{self.save_dir}"
        model_path = f"{path}_env_model.pth"
        model_error_path = f"{path}_env_model_error.pth"
        if hasattr(self, 'forward_model'):
            model_state_dict = {'forward_model': self.forward_model.state_dict()}
        if hasattr(self, 'error_model'):
            error_state_dict = {'error_model': self.error_model.state_dict()}
        torch.save(model_state_dict, model_path)
        torch.save(error_state_dict, model_error_path) 
        #print(f"Model saved to {path}")

    @classmethod
    def load(cls, path, observation_space, action_space, hidden_dim, compute_expected_error=True):
        """
        Load the model from a zip file
        """
        model = cls(observation_space=observation_space, action_space=action_space, hidden_dim=hidden_dim, compute_expected_error=compute_expected_error)
        model_path = f"{path}_env_model.pth"
        model_state_dict = torch.load(model_path)
        if compute_expected_error:
            model_error_path = f"{path}_env_model_error.pth"
            error_state_dict = torch.load(model_error_path)
        if hasattr(model, 'forward_model'):
            model.forward_model.load_state_dict(model_state_dict['forward_model'])
        if compute_expected_error and hasattr(model, 'error_model'):
            model.error_model.load_state_dict(error_state_dict['error_model'])
        print(f"Model loaded from {path}")
        return model
    
class InverseModelLearner(nn.Module):
    def __init__(self, 
                 observation_space, 
                 action_space, 
                 learning_rate=1e-7,
                 weight_decay=0.01,
                 hidden_dim=256, 
                 batch_size=64,
                 save_dir=None, 
                 normalize=False, 
                 ):
        super(InverseModelLearner, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.normalize = normalize
        self.observation_space = observation_space
        self.action_space = action_space
        self.batch_size = batch_size
        self.save_dir = save_dir
        if observation_space is not None:
            self.observation_size = observation_space.shape[0]
            self.action_size = action_space.shape[0]
            if normalize:
                self.obs_mean = torch.Tensor(observation_space.low + observation_space.high).to(self.device) / 2.0
                self.obs_std = torch.Tensor(observation_space.high - observation_space.low).to(self.device) / 2.0
                self.act_mean = torch.Tensor(action_space.low + action_space.high).to(self.device) / 2.0
                self.act_std = torch.Tensor(action_space.high - action_space.low).to(self.device) / 2.0
            self.inverse_model = self.generate_nn_architecture(hidden_dim=hidden_dim)
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
            self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.96)
            self.loss_fn = nn.MSELoss()
            self.loss = None

    def generate_nn_architecture(self, hidden_dim=256):
        """
        Generate the neural network architecture to model each sensor type
        params:
            hidden_dim: the number of hidden units in each hidden layer
        """
        network = nn.Sequential(
            nn.Linear(2 * self.observation_space.shape[0], hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, self.action_space.shape[0])
        )
        return network.to(self.device)

    def normalize_obs(self, obs):
        obs = (obs - self.obs_mean) / self.obs_std
        return obs

    def normalize_act(self, act):
        if isinstance(self.action_space, gym.spaces.MultiDiscrete):
            act = gym.spaces.flatten(self.action_space, act)
            act = torch.Tensor([((a - m) / s) for a, m, s in zip(act, self.act_mean, self.act_std)])
        else:
            act = (act - self.act_mean) / self.act_std
        return act

    def denormalize_obs(self, obs_norm):
        obs = (obs_norm * self.obs_std) + self.obs_mean
        return obs

    def denormalize_act(self, act_norm):
        if isinstance(self.action_space, gym.spaces.MultiDiscrete):
            act_norm = act_norm.tolist()
            act = gym.spaces.unflatten(self.action_space, act_norm)
            act = [a * s + m for a, m, s in zip(act, self.act_mean, self.act_std)]
            act = torch.Tensor(act)
        else:
            act = (act_norm * self.act_std) + self.act_mean
        return act

    def forward(self, observation, next_observation):
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if next_observation.ndim == 1:
            next_observation = next_observation.unsqueeze(0)

        # Pass the vector through the network
        x = self.inverse_model(torch.cat([observation, next_observation], dim=1))

        return x.to(self.device)

    def train(self, replay_data, num_epochs=1, validation=False):
        """
        Train the SAC agent on a dataset buffer
        """
        # Convert the observations, actions, and next observations to the appropriate device and data type
        observations = replay_data.observations.float().to(self.device)
        actions = replay_data.actions.float().to(self.device)
        next_observations = replay_data.next_observations.float().to(self.device)

        # Normalize the observations, actions, and next observations if normalization is enabled
        if self.normalize:
            observations = self.normalize_obs(observations)
            actions = self.normalize_act(actions)
            next_observations = self.normalize_obs(next_observations)

        # Initialize variables for validation if validation is enabled
        if validation:
            best_mse = float('inf')
            patience_counter = 0
            data = TensorDataset(observations, actions, next_observations)
            validation_data = random_split(data, [int(0.8*len(data)), int(0.2*len(data))])

        # Start training for the specified number of epochs
        for epoch in range(num_epochs):
            epoch_loss = 0.0

            # Zero the gradients of the optimizer
            self.optimizer.zero_grad()

            # Forward pass through the model
            pred_next_obs = self.forward(observations, actions)

            # Compute the loss
            sensor_loss = self.loss_fn(pred_next_obs, next_observations)
            epoch_loss += sensor_loss.item()

            # Compute the average loss and error loss
            epoch_loss /= len(observations)

            # Update the loss and error loss of the model
            self.loss = epoch_loss

            # Backpropagate the loss
            sensor_loss.backward(retain_graph=True)

            # Update the parameters of the model
            self.optimizer.step()
            self.scheduler.step()

            # Validate the model on the validation set if validation is enabled
            if validation:
                with torch.no_grad():
                    val_loss = 0.0

                    # Compute the validation loss and error loss
                    for obs, act, next_obs in validation_data[1]:
                        obs, act, next_obs = obs.to(self.device), act.to(self.device), next_obs.to(self.device)
                        pred_next_obs = self.forward(obs, act)
                        sensor_loss = self.loss_fn(pred_next_obs, next_obs)
                        val_loss += sensor_loss.item()

                    # Compute the average validation loss and error loss
                    val_loss /= len(validation_data[1])

                    # Update the best MSE and reset the patience counter if the validation loss is less than the best MSE
                    if val_loss < best_mse:
                        best_mse = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    # Stop training if the patience counter exceeds a certain limit
                    if patience_counter >= 10:
                        print("Early stopping")
                        break
                    print(f"Epoch {epoch} - Validation Loss: {val_loss}")

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureExtractor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.network(x)
    
class ICM(nn.Module):
    def __init__(self, observation_space, action_space, feature_dim, eta=0.2):
        super(ICM, self).__init__()
        self.feature_extractor = FeatureExtractor(observation_space.shape[0], feature_dim)
        self.forward_model = ForwardModelLearner(feature_dim, action_space)
        self.inverse_model = InverseModelLearner(feature_dim, action_space)
        self.eta = eta  # Scaling factor for intrinsic reward

    def forward(self, state, action, next_state):
        # Extract features from the states
        state_features = self.feature_extractor(state)
        next_state_features = self.feature_extractor(next_state)

        # Predict the next state features using the forward model
        predicted_next_state_features = self.forward_model(state_features, action)

        # Predict the action using the inverse model
        predicted_action = self.inverse_model(state_features, next_state_features)

        # Calculate the forward loss (prediction error)
        forward_loss = F.mse_loss(predicted_next_state_features, next_state_features)

        # Calculate the inverse loss (action prediction error)
        inverse_loss = F.mse_loss(predicted_action, action)

        # Calculate the intrinsic reward
        intrinsic_reward = self.eta * forward_loss.detach()

        return forward_loss, inverse_loss, intrinsic_reward