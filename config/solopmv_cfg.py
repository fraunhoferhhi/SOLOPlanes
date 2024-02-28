from modules.solopmv import SOLOP
from dotmap import DotMap

config = DotMap({
    'version': 'solopmv',
    'seed': 42,
    'backbone': {
        'name': 'resnet50',
        'use_pretrained': True,
        'path': './pretrained/resnet50.pth',
        'type': 'ResNetBackbone',
        'num_stages': 4,
        'frozen_stages': 1,
        'out_indices': (0, 1, 2, 3)
    },
    'neck': {
        'out_channels': 256,
    },
    'model':{
        'name': 'solop', 
        'init': SOLOP,
        'init_pretrained_solo_path': "", 
        'path':"", 
        'num_classes': 41,
        'mv_mode': True,
        'use_plane_feat_head': True,
        'use_same_feat_lvls': True,
        'cate_loss_weight': 1,
        'dice_loss_weight': 3,
        'mask_feat_channels': 128,
        'plane_head_kernel': 3,
        'plane_feat_channels': 64, 
        'plane_feat_xycoord': False,
        'planefeat_startlvl': 0,
        'planefeat_endlvl': 1,
    },
    'dataset': {
        'datafolder': '/mnt/data/datasets/ScanNet',
        'train_subset':  30000,
        'val_subset':  5000,
        'augment': False,
        'MAX_DEPTH': 10,
        'original_size': (480,640), 
        'input_size':  (480,640), 
        'img_mean': [0.485, 0.456, 0.406], 
        'img_std': [0.229, 0.224, 0.225], 
        'depth_error_margin':  0.10, 
        'mv': dict(views = 2, steps_between_views = 10)
        },
    'test_cfg': dict(
                nms_pre=500,
                score_thr=0.1,
                mask_thr=0.5,
                update_thr=0.05,
                kernel='gaussian',  # gaussian/linear
                sigma=2.0,
                max_per_img=40),
    'lr': 1e-4,
    'start_lr': 1e-8,
    'lr_warmup_steps':500,
    'optimizer': torch.optim.AdamW,
    'epochs': 20,
    'batch_size': 16,
    'shuffle': True,
    'num_workers': 8,
    # 'limit_train_batches': 0.05,
    # 'limit_val_batches': 0.05,
    'accumulate_grad_batches': 4,
    })


config.BASE_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '../'))
config.dataset.train_indices = config.BASE_PATH + "/train_mv_indices.pkl"
config.dataset.val_indices = config.BASE_PATH + "/val_mv_indices.pkl"
config.dataset.test_indices = config.BASE_PATH + "/planemvs_test_indices.pkl"

