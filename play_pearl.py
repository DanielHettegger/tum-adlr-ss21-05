import gym
import numpy as np
from stable_baselines3.common.callbacks import EventCallback
import torch as th

import kuka_iiwa_insertion

from stable_baselines3 import PEARL


def play():
    env = gym.make('kuka_iiwa_insertion-v0', use_gui=True)
    model = PEARL.load("models/feasible-dust-68_latest_model", env=env)

    model.callback = EventCallback()


    i = 0
    while True:
        task_idx = i%env.number_of_tasks
        reward = eval_task(model, model.env, task_idx)
        print("Task {} Average reward was {}".format(task_idx, reward))
        i+=1




def eval_task(model, env, task_idx):
    print("Evaluating model in env with task idx {}".format(task_idx))
    with th.no_grad():
        env.env_method("reset_task", task_idx)

        model.actor.clear_z()
        num_transitions = 0
        num_trajs = 0

        model.JUST_EVAL.reset()
        n_trajs = 10
        while num_transitions < 1000 * n_trajs and num_trajs <= n_trajs:
            num = model.obtain_samples(
                deterministic=True,
                max_samples=1000 ,#- num_transitions,
                max_trajs=1,
                accum_context=True,
                replaybuffers=[model.JUST_EVAL],
            )
            num_transitions += num
            num_trajs += 1
            model.actor.infer_posterior(model.actor.context)

        # Collect reward with infered posterior
        model.JUST_EVAL.reset()

        print("Posterior initialized. Now collecting reward data.")

        num_transitions = 0
        num_trajs = 0
        n_trajs = 3
        while num_transitions < 1000 * n_trajs and num_trajs <= n_trajs:
            num = model.obtain_samples(
                deterministic=True,
                max_samples=1000 ,#- num_transitions,
                max_trajs=1,
                accum_context=True,
                replaybuffers=[model.JUST_EVAL],
            )
            num_transitions += num
            num_trajs += 1


        rwd = model.JUST_EVAL.rewards[range(model.JUST_EVAL.pos)]
        return (np.sum(rwd) / n_trajs)

if __name__ == "__main__":
    play()