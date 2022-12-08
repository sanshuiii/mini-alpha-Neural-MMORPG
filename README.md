# Mini Alpha Neural MMORPG

## Introduction
This is a README file for our mini alpha Neural MMORPG program. In this project, codes are split into several parts. ```/ppo``` gives an example of distributed version of [PPO](https://arxiv.org/abs/1707.06347) algorithm. 

- ```/ppo/model``` is the structure of out policy network, which consists of a main body (```Arch.py```), Encoders, LSTM core and action heads.
- ```/ppo/rollouts``` is the data rollout.
- ```/ppo/scripts/config``` holds the config files.
- ```/ppo/utils``` holds some utils.
- ```/ppo/PPO_discrete_main.py``` is a discrete version of PPO, while ```/ppo/PPO_discrete_main_dist.py``` is a distributed version (implemented with ```torch.distributed.rpc```).
- ```/ppo/SupervisedLearning.py``` is the imitating training code.

### Our model
o implement a multi-agent policy, we design a hierarchical structure consisting of a series of deep
models. For the front end of the model, we aggregate observations of different agents and take the
advantage of resnet and transformer to encode the observations from both spatial and substantial
aspects. After that, the encoded data will be fed into an LSTM to be combined with the historical data.
And finally, action heads will decide according to all information mentioned above.

[architecture.png](architecture.png)

The whole training process consists of two stages: supervised learning and reinforcement learning. In
the supervised learning stage, we train the model using game replays of rule-based agents to obtain a
base model that can imitate rule-based agents. Then the pre-trained model will be put into a self-play
scenario, using the reinforcement learning method to continuously adjust the policy network.

## Quick Start

First git clone from [Neural MMO Starter Kit](https://gitlab.aicrowd.com/neural-mmo/neurips2022-nmmo-starter-kit)
```
git clone http://gitlab.aicrowd.com/neural-mmo/neurips2022-nmmo-starter-kit.git
conda create -n neurips2022-nmmo python==3.9
conda activate neurips2022-nmmo
cd ./neurips2022-nmmo-starter-kit
```

Then install git-lfs and download the environment wrapper prepared for NeurIPS 2022 Competition.
```
apt install git-lfs
pip install git+http://gitlab.aicrowd.com/neural-mmo/neurips2022-nmmo.git
pip install -r requirements_tool.txt
```

You can try to submit with
```
python tool.py submit "my-first-submission"
```

Run local evalution with
```
from neurips2022nmmo import CompetitionConfig, scripted, submission, RollOut

config = CompetitionConfig()

my_team = submission.get_team_from_submission(
    submission_path="my-submission/",
    team_id="MyTeam",
    env_config=config,
)
# Or initialize the team directly
# my_team = MyTeam("Myteam", config, ...)

teams = [scripted.CombatTeam(f"Combat-{i}", config) for i in range(5)]
teams.extend([scripted.MixtureTeam(f"Mixture-{i}", config) for i in range(10)])
teams.append(my_team)

ro = RollOut(config, teams, parallel=True, show_progress=True)
ro.run(n_episode=1)
```

Move this model to 
