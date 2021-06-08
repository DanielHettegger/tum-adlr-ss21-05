import gym
import numpy as np
import matplotlib.pyplot as plt
import os

import kuka_iiwa_insertion


from stable_baselines3 import SAC

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import get_log_dict

import wandb


def main():
    # 1. Start a W&B run
    wandb.init(project='pearl', entity='adlr-ss-21-05')
    print("wandb name: ", wandb.run.name)
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)
    callback = SaveOnBestTrainingRewardCallback(
        check_freq=1000,check_log=1, log_dir=log_dir)

    env = gym.make('kuka_iiwa_insertion-v0', use_gui=False)
    env = Monitor(env, log_dir)

    try:
        model = SAC.load("models/kuka_iiwa_insertion-v0_sac_best_model",
                         env, verbose=1, learning_starts=1000)
    except:
        model = SAC("MlpPolicy", env, verbose=1)

    i = 0
    save_interval = 10000
    while True:
        i += save_interval
        model.learn(total_timesteps=save_interval, callback=callback)
        print("saving model {}".format(i))
        model.save("models/kuka_iiwa_insertion-v0_sac")
        #plot_results([log_dir], save_interval, results_plotter.X_TIMESTEPS, "SAC Insertion")
        # plt.show()


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).
    Additionally logs actor critic and reward to weights and biases 

    :param check_freq: (int)
    :check_log: (int) 
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, check_log: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.check_log = check_log
        self.log_dir = log_dir
        self.save_path = 'models'
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                        self.best_mean_reward, mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(os.path.join(
                        self.save_path, 'kuka_iiwa_insertion-v0_sac_best_model'))
        

        return True
    
    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        dict = get_log_dict()
        actor_loss = dict.get("train/actor_loss")
        critic_loss = dict.get("train/critic_loss")
        if actor_loss:
            wandb.log({"actor_loss": actor_loss})
        if critic_loss:
            wandb.log({"critic_loss": critic_loss})

        x, y = ts2xy(load_results(self.log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            wandb.log({"episode_reward": y[-1]})
            wandb.log({"mean_episode_reward": y[-1]})
        


if __name__ == '__main__':
    main()
