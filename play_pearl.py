import gym
import numpy as np
from stable_baselines3.common.callbacks import EventCallback
import torch as th
import sys
import argparse


import kuka_iiwa_insertion

from stable_baselines3 import PEARL


def play(args):
    env = gym.make('kuka_iiwa_insertion-v0', use_gui=True)
    model = PEARL.load(args.model, env=env)

    model.callback = EventCallback()

    if args.disable_latent:
        model.actor.use_latent = 0

    no_tasks = args.no_tasks if args.no_tasks is not -1 else env.number_of_tasks

    i = 0
    while True:
        task_idx = i % no_tasks
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="models/dashing-butterfly-85_best_model")
    parser.add_argument("-n","--no-tasks", type=int, default=-1)
    parser.add_argument('-d','--disable-latent', dest='disable_latent', action='store_true')

    play(parser.parse_args())