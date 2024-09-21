### Installation

```sh
conda create --name $yourEnv python=3.8
conda activate $yourEnv
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/openai/CLIP.git
python -m pip install -r requirements.txt
mkdir ./atmodel_data
wget -P ./xdecoder_data https://huggingface.co/xdecoder/X-Decoder/resolve/main/coco_caption.zip
unzip ./xdecoder_data/coco_caption.zip -d ../xdecoder_data
```
