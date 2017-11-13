import argparse
import gym
from gym import wrappers
import os.path as osp
import os
import time
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from multiprocessing import Process
import dqn_log as dqn
from dqn_utils import *
from atari_wrappers import *


def atari_model(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out

def atari_learn(exp_name,
                task,
                seed,
                logdir,
                checkpoint_dir,
                num_timesteps,
                target_update_freq):

    # get environment
    env = get_env(task, seed)
    session = get_session()
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0
    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (10e6/2, 0.01),
        ], outside_value=0.01
    )

    dqn.learn(
        exp_name,
        env,
        q_func=atari_model,
        optimizer_spec=optimizer,
        session=session,
        logdir = logdir,
        checkpoint_dir = checkpoint_dir,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=target_update_freq,
        grad_norm_clipping=10
    )
    env.close()

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i) 
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session

def get_env(task, seed):
    env_id = task.env_id

    env = gym.make(env_id)

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/hw3_vid_dir2/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)

    return env

def main(args):

    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')
    # Change the index to select a different game.
    task = benchmark.tasks[3]
    env_name = task.env_id

    # create the logs directory
    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)
        # Run training
        num_timesteps = args.num_timesteps or task.max_timesteps
        def train_func():
            atari_learn(args.exp_name,
                        task,
                        seed, 
                        logdir=os.path.join(logdir,'%d'%seed),
                        checkpoint_dir=args.checkpoint_dir ,
                        num_timesteps=num_timesteps,
                        target_update_freq=args.target_update_freq)

        p = Process(target=train_func, args=tuple())
        p.start()
        p.join()



def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)


    parser.add_argument(
        "--target_update_freq",
        type=int,
        default=10000,
        help="How often to copy current network to target network",
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
	    default=16000000,
        help="Maximum number of timesteps to run",
    )
    parser.add_argument(
        "--replay_buffer_size",
        type=int,
        default=1000000,
        help="Size of the replay buffer",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default='./checkpoints',
        help="Directory to checkpoint NN",
    )

    return parser


if __name__ == "__main__":
    main(get_arg_parser().parse_args())
