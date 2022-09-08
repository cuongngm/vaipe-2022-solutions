pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
mkdir checkpoints
cd checkpoints
# detect pill model
gdown --id 11a9N-7AOG5QcswCtvqj1CVdgHWRjzr5i
# detect drugname model
gdown --id 1kXMmG6aqRyFAAo2Kr_GmpPL696s6r1i0
# recog drugname model
gdown --id 1itEm13CQQ8szInHgPKsVpC860sonTCO6
# recog pill model
# extract feature pill model
gdown --id 1dclf-hsblbzUwXzMAW-N76GxkgNMzTVN
cd ../
