import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

import kuka_iiwa_insertion


from stable_baselines3 import SAC

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import get_log_dict

import wandb

def main(args):
    # 1. Start a W&B run
    wandb.init(project='pearl', entity='adlr-ss-21-05')
    wandb.config.update(args)
    print("wandb name: ", wandb.run.name)
    
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)
    
    callback = SaveOnBestTrainingRewardCallback(
        check_freq=1000,check_log=1, log_dir=log_dir, model_name=wandb.run.name)

    env = gym.make('kuka_iiwa_insertion-v0', use_gui=False, steps_per_action=args.steps_per_action, max_steps=args.max_steps, action_step_size=args.action_step_size)
    env = Monitor(env, log_dir)

    model = SAC("MlpPolicy", env, verbose=args.verbosity, train_freq=(args.train_freq_num, args.train_freq_type), batch_size=args.batch_size)

    i = 0
    save_interval = 1000000
    while True:
        i += save_interval
        model.learn(total_timesteps=save_interval, callback=callback)


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

    def __init__(self, check_freq: int, check_log: int, log_dir: str, model_name: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.check_log = check_log
        self.log_dir = log_dir
        self.save_path = 'models'
        self.model_name = model_name
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
                        self.save_path, self.model_name + '_best_model'))
                    self.model.save(os.path.join(
                        self.save_path,  'kuka_iiwa_insertion-v0_sac_best_model'))
        

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
            wandb.log({"episode_reward": y[-1]})

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", type=int, default=1)
    parser.add_argument("-g", "--git_commit")
    parser.add_argument("-s", "--max_steps", type=int, default=1000)
    parser.add_argument("--action_step_size", type=float, default=0.005)
    parser.add_argument("--steps_per_action", type=int, default=1)    
    parser.add_argument("--train_freq_num", type=int, default=1)    
    parser.add_argument("--train_freq_type", type=str, default="episode", choices=["episode","step"]) 
    parser.add_argument("--batch_size", type=int, default=256)    

    main(parser.parse_args())
