import numpy as np
from torch.utils.tensorboard import SummaryWriter
from NeurIPS22NMMO.ppo.utils.normalization import Normalization, RewardScaling
from NeurIPS22NMMO.ppo.utils.replaybuffer import ReplayBuffer
from ppo_discrete import PPO_discrete

import os
import time
import argparse
import gym
import threading

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from torch.distributed.rpc import RRef, remote

import yaml
from easydict import EasyDict

NUM_STEPS = 500
AGENT_NAME = "agent"
OBSERVER_NAME = "observer{}"

def evaluate_policy(args, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:  # During the evaluating,update=False
            s = state_norm(s, update=False)
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            s_, r, done, _ = env.step(a)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times

class Agent:
    def __init__(self, world_size, args, env_name, seed=0):
        self.world_size = world_size
        self.args = args
        self.env_name = env_name
        self.seed = seed

        self.init_seed(seed)        
        self.init_clients()
        self.init_writer()
        self.init_model()
        self.init_eval_env()

    def init_seed(self, seed):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)


    def init_clients(self):
        self.ob_rrefs = []
        self.agent_rref = RRef(self)

        for ob_rank in range(1, self.world_size):
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(ob_rank))
            self.ob_rrefs.append(remote(ob_info, Observer, args=(self.env_name, self.seed)))
    
    def reset_env_and_state(self,):
        futs = []
        for ob_rref in self.ob_rrefs:
            futs.append(ob_rref.rpc_async().reset())
        rets = torch.futures.wait_all(futs)
        self.states = np.array([ret for ret in rets])

    def init_writer(self):
        self.writer = SummaryWriter(log_dir='/hdd/yifan/nips22nmmo/NeurIPS22NMMO/ppo/runs/PPO_discrete/env_{}_number_{}_seed_{}'.format(self.env_name, 1, self.seed))

    def init_model(self):
        self.agent = PPO_discrete(self.args)    
        self.state_norm = Normalization(shape=self.args.state_dim)  # Trick 2:state normalization
        if self.args.use_reward_norm:  # Trick 3:reward normalization
            self.reward_norm = Normalization(shape=1)
        elif self.args.use_reward_scaling:  # Trick 4:reward scaling
            self.reward_scaling = RewardScaling(shape=1, gamma=self.args.gamma)
    
    def init_eval_env(self,):
        self.env_evaluate = gym.make(self.env_name)
        self.env_evaluate.reset(seed=self.seed)
        self.env_evaluate.action_space.seed(self.seed)
    
    def do_train(self, ):
        self.evaluate_num = 0 # Record the number of evaluations
        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.episode_steps = [0] * len(self.ob_rrefs)

        self.total_steps = 0  # Record the total steps during the training
        self.replay_buffer = ReplayBuffer(self.args)
        
        self.reset_env_and_state()
        
        while self.total_steps < self.args.max_train_steps:
            self.train_step()
            
    def evaluate_step(self, ):
        self.evaluate_num += 1
        evaluate_reward = evaluate_policy(self.args, self.env_evaluate, self.agent, self.state_norm)
        self.evaluate_rewards.append(evaluate_reward)
        print(f"evaluate_num:{self.evaluate_num} \t evaluate_reward:{evaluate_reward}")
        self.writer.add_scalar(f'step_rewards_{self.env_name}', self.evaluate_rewards[-1], global_step=self.total_steps)

        # Save the rewards
        if self.evaluate_num % self.args.save_freq == 0:
            np.save('/hdd/yifan/nips22nmmo/NeurIPS22NMMO/ppo/data/PPO_discrete_env_{}_number_{}_seed_{}.npy'.format(self.env_name, 1, self.seed), np.array(self.evaluate_rewards))
            
    def train_step(self):
        args = self.args
        replay_buffer = self.replay_buffer
        states = self.states
        actions, action_logprobs = self.agent.choose_action(states, single=False)
        
        futs = []
        for ob_rref, a in zip(self.ob_rrefs, actions):
            futs.append(ob_rref.rpc_async().env_step(a))
            if self.total_steps // args.batch_size // 16 < len(futs): # TODO: add to args
                pass #break
            
        if replay_buffer.count >= args.batch_size: # TODO: chunk the size if count > batch_size
            self.agent.update(replay_buffer, self.total_steps)
            replay_buffer.count = 0 # TODO: chunk the size if count > batch_size
        
        if (self.total_steps) % args.evaluate_freq < len(futs):#len(self.ob_rrefs):
            self.evaluate_step()
            
        rets = torch.futures.wait_all(futs)
        
        for idx, (s, a, a_logprob, (s_, r, done, _)) in enumerate(
            zip(states, actions, action_logprobs, rets)
        ):
            if args.use_state_norm:
                s_ = self.state_norm(s_)
            if args.use_reward_norm:
                r = self.reward_norm(r)
            elif args.use_reward_scaling:
                r = self.reward_scaling(r)
                
            if done and self.episode_steps[idx] < args.max_episode_steps:
                dw = True
            else:
                dw = False

            if replay_buffer.count < args.batch_size: # TODO chunk the size if count > batch_size
                self.replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
                self.total_steps += 1
            self.states[idx] = s_


class Observer:
    def __init__(self, env_name, seed):
        self.id = rpc.get_worker_info().id - 1
        self.env_name = env_name
        self.seed = seed + self.id
        
        self.init_seed(self.seed)
        self.init_env()
        
        self.future_state = torch.futures.Future()
        self.lock = threading.Lock()
        self.n_agent = 1
        self.pending_states = self.n_agent # TODO
        
    def init_seed(self, seed):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)

    def init_env(self):
        self.env = gym.make(self.env_name)
        self.env.reset(seed=self.seed)#self.env.seed(self.seed)
        self.env.action_space.seed(self.seed)
        
    def reset(self):
        return self.env.reset()
    
    def env_step(self, a):
        s_, r, done, _ = self.env.step(a)
        if done:
            s_ = self.env.reset()
        return s_, r, done, _
        
    
def run_worker(rank, world_size, args, env_name, seed):
    r"""
    This is the entry point for all processes. The rank 0 is the agent. All
    other ranks are observers.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    env = gym.make(env_name)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.n
    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    
    if rank == 0:
        print("env={}".format(env_name))
        print("state_dim={}".format(args.state_dim))
        print("action_dim={}".format(args.action_dim))
        print("max_episode_steps={}".format(args.max_episode_steps))
    
    if rank == 0:
        # rank0 is the agent
        rpc.init_rpc(AGENT_NAME, rank=rank, world_size=world_size, 
                    rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                        num_worker_threads=8,
                        _transports=["uv"] #["shm", "uv"] # forbid shm
                    ))

        agent = Agent(world_size, args, env_name, seed)
        agent.do_train()
    else:
        rpc.init_rpc(OBSERVER_NAME.format(rank), rank=rank, world_size=world_size)
    rpc.shutdown()

def launch(n_obs, args, env_name='CartPole-v1', seed=0):
    world_size = n_obs + 1
    print(f"###################### world_size: {world_size} ######################")
    delays = []
    tik = time.time()
    mp.spawn(
        run_worker,
        args=(world_size, args, env_name, seed),
        nprocs=world_size,
        join=True
    )
    tok = time.time()
    delays.append(tok - tik)
    print(f"{world_size}, {delays[-1]}")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    # parser.add_argument("--max_train_steps", type=int, default=int(2e5), help=" Maximum number of training steps")
    # parser.add_argument("--evaluate_freq", type=int, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    # parser.add_argument("--save_freq", type=int, default=10, help="Save frequency")
    # parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    # parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    # parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    # parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    # parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    # parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    # parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    # parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    # parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    # parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    # parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    # parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    # parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    # parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    # parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    # parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    # parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    # parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    # parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    #
    # args = parser.parse_args()
    #
    # print(args)

    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--config_path", default="./scripts/config/config.yml")
    with open(parser.parse_args().config_path, 'r') as f:
        arg_file = yaml.full_load(f)
    args = EasyDict(arg_file)


    n_obs = 8
    launch(n_obs, args, env_name='CartPole-v1', seed=0)