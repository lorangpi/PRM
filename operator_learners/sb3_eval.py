from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

class FixedSeedEvalCallback(EvalCallback):
    """
    Modify EvalCallback such that a fixed seed is set after each round of evaluations
    and that best models are stored with specified filename.
    """

    def __init__(self, *args, seed=None, initial_eval=True, **kwargs):
        self.seed = seed
        super().__init__(*args, **kwargs)

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            print("Setting fixed seed for evaluations ...")
            self.eval_env.seed(self.seed)
        super()._on_step()
