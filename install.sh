source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -n actionformer python=3.9 -y
conda activate actionformer
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install tensorboard
pip install numpy==1.22.4
pip install pyyaml pandas h5py joblib
pip install opencv-python==4.11.0.86
cd actionformer_release/libs/utils
python setup.py install --user
cd ../../..


conda deactivate
conda create -n openmmlab python=3.9 -y
conda activate openmmlab
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -U openmim
mim install mmengine
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3/index.html
mim install "mmdet==3.2.0"
sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" $CONDA_PREFIX/lib/python3.9/site-packages/mmdet/__init__.py
cd mmpose
pip install -e .

