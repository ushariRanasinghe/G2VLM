
git clone https://github.com/InternRobotics/G2VLM
cd G2VLM
# conda create -n g2vlm python=3.10 -y
# conda activate g2vlm

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

#Download model weights
python -c "from huggingface_hub import snapshot_download; \
save_dir='models/G2VLM-2B-MoT'; \
repo_id='InternRobotics/G2VLM-2B-MoT'; \
cache_dir=save_dir+'/cache'; \
snapshot_download(cache_dir=cache_dir, local_dir=save_dir, repo_id=repo_id, local_dir_use_symlinks=False, resume_download=True, allow_patterns=['*.json','*.safetensors','*.bin','*.py','*.md','*.txt'])"

#Fix for missing __init__.py files
touch /workspace/G2VLM/modeling/pi3/__init__.py
touch /workspace/G2VLM/modeling/pi3/utils/__init__.py

#add pi3 package to PYTHONPATH
cd G2VLM
export PYTHONPATH=/workspace/G2VLM/modeling:$PYTHONPATH

#install additional dependencies
pip install easydict 
pip install --ignore-installed blinker
pip install open3d

python inference_recon.py --model_path /workspace/models/G2VLM-2B-MoT

# use python inference_recon.py --model-path /workspace/models/G2VLM-2B-MoT to run the spatial reasoning demo
