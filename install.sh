conda create -n mmo python==3.9
conda activate  mmo
cd ./neurips2022-nmmo-starter-kit
pip install git+http://gitlab.aicrowd.com/neural-mmo/neurips2022-nmmo.git
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements_tool.txt
pip install gym==0.23.1 pyyaml easydict pygame jupyter
