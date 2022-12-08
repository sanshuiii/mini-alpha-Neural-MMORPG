# mini-alpha-Neural-MMORPG

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
