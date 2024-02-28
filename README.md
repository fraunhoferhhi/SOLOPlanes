# SOLOPlanes

This is the official implementation of [Multi-task Planar Reconstruction with Feature Warping Guidance](https://arxiv.org/abs/2311.14981).

## Dataset
Download the [ScanNet](http://www.scan-net.org/ScanNet/) dataset and follow instructions to extract sens data for RGB, depth, and camera parameters. 

We use plane annotations from https://github.com/NVlabs/planercnn#training  

The data directory should look like:

    ScanNet
    ├── scans
    │   └── sceneXXXX_XX
    │       ├── annotation 
    │       ├── sens    
    │       └── ...
    └── ...



##  Getting started
```
conda create -n testsolop python=3.10
pip install -r requirements.txt

```

## Training
Download pretrained backbones [here](url) and put in `pretrained/` folder

Edit config files with correct dataroot and settings

`python train_local.py`

## Evaluation
Download the desired pretrained model [here](url) 

Update the corresponding config file model.path

`python eval_models.py` 





## Acknowledgement
- _Modules adjusted from: https://github.com/OpenFirework/pytorch_solov2_
- _ScanNet dataset annotations from: https://github.com/NVlabs/planercnn_
