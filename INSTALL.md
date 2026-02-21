### Create venv

```bash
apt-get update && apt-get install -y curl bzip2

curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

mv bin/micromamba /usr/local/bin/micromamba

micromamba shell init -s bash /root/micromamba
source ~/.bashrc

git clone https://github.com/nvnhat95/sam-3d-objects.git
cd sam-3d-objects
micromamba env create -f environments/default.yml
micromamba activate sam3d-objects
```

### Download checkpoint
Note: accepted the agreement on Sam3D HF repo, then login with 
```bash
pip install 'huggingface-hub[cli]<1.0'
hf auth login
```

```bash
TAG=hf
hf download \
  --repo-type model \
  --local-dir checkpoints/${TAG}-download \
  --max-workers 1 \
  facebook/sam-3d-objects
mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download
```

### Installation
```bash
# for pytorch/cuda dependencies
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"

# install sam3d-objects and core dependencies
pip install -e '.[dev]'
pip install -e '.[p3d]' # pytorch3d dependency on pytorch is broken, this 2-step approach solves it

# for inference
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'

# patch things that aren't yet in official pip packages
./patching/hydra # https://github.com/facebookresearch/hydra/pull/2863
```

### Download sample data
```bash
mkdir data && cd data
wget https://www.dropbox.com/scl/fi/2hovd64bmaemiby52bqx0/lego.zip?rlkey=k9ocih0r86x93hval9jh1emmm&st=26gm89o9&dl=1
mv lego.zip* lego.zip
apt-get install unzip
unzip lego.zip
cd ..
```

### Run
```bash
python sam_3d_consistent.py --config_path checkpoints/hf/pipeline.yaml --blender_dir data/lego/ --mask_pt data/lego/mask_train.pt --freeze_backbone
```