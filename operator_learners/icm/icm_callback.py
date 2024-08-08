from stable_baselines3.common.callbacks import BaseCallback

class ICMTrainingCallback(BaseCallback):
    def __init__(self, icm, verbose=0):
        super().__init__(verbose)
        self.icm = icm

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:  # Train every 1000 steps
            replay_data = self.model.replay_buffer.sample(1000)  # Sample 1000 transitions from the replay buffer
            self.icm.train(replay_data, num_epochs=1)  # Train the ICM model and get the loss

            # Log the ICM loss
            self.logger.record('train/icm_forward_loss', self.icm.forward_loss)
            self.logger.record('train/icm_inverse_loss', self.icm.inverse_loss)

        return True