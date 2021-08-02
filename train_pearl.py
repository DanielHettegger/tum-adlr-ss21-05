import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time

import kuka_iiwa_insertion
from numpy.core.fromnumeric import mean


from stable_baselines3 import PEARL

from stable_baselines3.common.logger import Logger,CSVOutputFormat
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_meta_policy
#from stable_baselines3.common.logger import get_log_dict

import wandb

def main(args):
    print(args)
    # 1. Start a W&B run
    if not args.no_logging:
        wandb.init(project='pearl', entity='adlr-ss-21-05')
        wandb.config.update(args)
        print("wandb name: ", wandb.run.name)
        run_name = wandb.run.name
    else:
        run_name = "run_" + str(time.time())

    log_dir = "tmp/" + run_name + "/"
    os.makedirs(log_dir, exist_ok=True)
    
    callback = SaveOnBestTrainingRewardCallback(
        check_freq=1000,check_log=1, log_dir=log_dir, model_name=run_name, wandb_logging= not args.no_logging)

    env = gym.make('kuka_iiwa_insertion-v0', use_gui=False, steps_per_action=args.steps_per_action, max_steps=args.max_steps, action_step_size=args.action_step_size)
    env = Monitor(env, log_dir)

    no_tasks = args.no_tasks if args.no_tasks is not -1 else env.number_of_tasks

    model = PEARL("MlpPolicy", env, 
        verbose=args.verbosity, 
        learning_rate=args.learning_rate,
        train_freq=(args.train_freq_num, args.train_freq_type), 
        batch_size=args.batch_size,
        n_traintasks = no_tasks,
        n_evaltasks = no_tasks,
        n_epochtasks= no_tasks,
        latent_dim = args.latent_dim)

    if args.disable_latent:
        model.actor.use_latent = 0
    
    model.set_logger(Logger(log_dir, [CSVOutputFormat(log_dir+"log.csv")]))

    i = 0
    eval_interval = 10000
    prev_mean_rew = -np.inf
    while True:
        i += eval_interval
        model.learn(total_timesteps=eval_interval, callback=callback)
        mean_reward, std_reward = evaluate_meta_policy(model, model.env)
        if not args.no_logging:
            wandb.log({"eval_mean_reward":mean_reward})
        if mean_reward > prev_mean_rew:
            prev_mean_rew = mean_reward
            model.save(os.path.join(
                            "models", run_name + '_best_model'))
        model.save(os.path.join(
                            "models", run_name + '_latest_model'))


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

    def __init__(self, check_freq: int, check_log: int, log_dir: str, model_name: str, verbose=1, wandb_logging=True):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.check_log = check_log
        self.log_dir = log_dir
        self.save_path = 'models'
        self.model_name = model_name
        self.best_mean_reward = -np.inf
        self.wandb_logging = wandb_logging

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        kl_div = self.locals.kl_div.item()
        kl_loss = self.locals.kl_loss.item()
        critic_loss = self.locals.critic_loss.item()
        actor_loss = self.locals.actor_loss.item()

        if self.wandb_logging:
            wandb.log({"kl_div": kl_div, "kl_loss":kl_loss, "critic_loss":critic_loss, "actor_loss":actor_loss})

        return True
    
    def _on_rollout_end(self) -> None:
        mean_reward = self.locals["mean_reward"]
        episode_rewards = self.locals["episode_rewards"]
        if self.wandb_logging:
            wandb.log({"episode_reward": episode_rewards[-1], "mean_episode_reward":mean_reward})# "actor_loss": actor_loss, "critic_loss": critic_loss})

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", type=int, default=1)
    parser.add_argument("-s", "--max_steps", type=int, default=1000)
    parser.add_argument("--action_step_size", type=float, default=0.005)
    parser.add_argument("--steps_per_action", type=int, default=1)    
    parser.add_argument("--train_freq_num", type=int, default=1)    
    parser.add_argument("--train_freq_type", type=str, default="episode", choices=["episode","step"]) 
    parser.add_argument("--batch_size", type=int, default=256)    
    parser.add_argument('--no-logging', dest='no_logging', action='store_true')
    parser.add_argument('-n',"--no-tasks", type=int, default=-1)
    parser.add_argument("--latent-dim", type=int, default=5)
    parser.add_argument('--disable-latent', dest='disable_latent', action='store_true')
    parser.add_argument('-l',"--learning-rate", type=float, default=0.001)



    main(parser.parse_args())
