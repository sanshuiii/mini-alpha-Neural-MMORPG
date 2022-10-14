import torch
import numpy as np
import nmmo
from nmmo import config
from neurips2022nmmo import CompetitionConfig

class TrainConfig(CompetitionConfig):
    MAP_N = 1

conf = TrainConfig()
env = nmmo.Env( conf )

a = env.observation_space(agent=1)
print(a)

obs = env.reset()
print(obs.keys())

print(obs[1].keys())

print(obs[1]['Entity'].keys())