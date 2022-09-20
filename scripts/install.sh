pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
mkdir checkpoints
cd checkpoints
# detect drugname model
gdown --id 1kXMmG6aqRyFAAo2Kr_GmpPL696s6r1i0
# recog drugname model
gdown --id 1itEm13CQQ8szInHgPKsVpC860sonTCO6
# recog pill model
# dict mapping don thuoc
gdown --id 1a5gTKsBqrTmg17x2Diz5mo-AoYXGkXwm
# detect pill model
gdown --id 1yy7LMOOhot2EQ1Bq4S1XbPVyQNnUFE33
# extract feature pill model
gdown --id 15hvwXB-Vipllgieg-7aqUfPWNTdR8FDZ
gdown --id 1_j0HfU14JuD45A2h1c1uhMtT8gA2WuPF
cd ../
