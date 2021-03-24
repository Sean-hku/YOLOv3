#-*-coding:utf-8-*-

cmds = [

# 'CUDA_VISIBLE_DEVICES=2 python train_sparse.py --type original --activation leaky --batch-size 16 --freeze False --epochs 100 --LR 0.00025 --optimize sgd --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale True --img_size 416 --rect True -sr --s 0.0003 --data data/rgb/rgb.data --expFolder swim0315_sp	--expID 5',
# 'CUDA_VISIBLE_DEVICES=2 python train_sparse.py --type original --activation leaky --batch-size 16 --freeze False --epochs 100 --LR 0.00025 --optimize sgd --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale True --img_size 416 --rect True -sr --s 0.0004 --data data/rgb/rgb.data --expFolder swim0315_sp	--expID 6',
# 'CUDA_VISIBLE_DEVICES=2 python train_sparse.py --type original --activation leaky --batch-size 16 --freeze False --epochs 100 --LR 0.00025 --optimize sgd --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale False --img_size 416 --rect True -sr --s 0.0005 --data data/rgb/rgb.data --expFolder swim0315_sp	--expID 7',
# 'CUDA_VISIBLE_DEVICES=2 python train_sparse.py --type original --activation leaky --batch-size 16 --freeze False --epochs 100 --LR 0.00025 --optimize sgd --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale False --img_size 416 --rect True -sr --s 0.0006 --data data/rgb/rgb.data --expFolder swim0315_sp	--expID 8',
#
# 'CUDA_VISIBLE_DEVICES=3 python train.py --type original --activation leaky --batch-size 16 --freeze False --epochs 200 --LR 0.00025 --optimize sgd --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale True --img_size 416 --rect True  --data data/rgb/rgb.data --expFolder swim0315	--expID 5',
# 'CUDA_VISIBLE_DEVICES=2 python train.py --type original --activation leaky --batch-size 16 --freeze False --epochs 100 --LR 0.00025 --optimize sgd --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale True --img_size 416 --rect True  --data data/rgb/rgb.data --expFolder swim0315	--expID 6',
# 'CUDA_VISIBLE_DEVICES=2 python train.py --type original --activation leaky --batch-size 16 --freeze False --epochs 100 --LR 0.00025 --optimize sgd --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale False --img_size 416 --rect True  --data data/rgb/rgb.data --expFolder swim0315	--expID 7',
# 'CUDA_VISIBLE_DEVICES=2 python train.py --type original --activation leaky --batch-size 16 --freeze False --epochs 100 --LR 0.00025 --optimize sgd --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale False --img_size 416 --rect True  --data data/rgb/rgb.data --expFolder swim0315	--expID 8',
'python train.py --type spp --activation leaky --batch-size 8 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/fish/fish.data --sr True --s 0.8 --expFolder testfirst	--expID 7',
# 'python train.py --type original --activation leaky --batch-size 8 --freeze False --prune 0 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/fish/fish.data --sr True --s 0.004 --expFolder testfirst	--expID 2',
# 'python train.py --type spp --activation leaky --batch-size 8 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize adam --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/fish/fish.data --sr False --expFolder testfirst	--expID 3',
# 'python train.py --type original --activation leaky --batch-size 8 --freeze False --prune 0 --epochs 100 --LR 0.00025 --optimize adam --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/fish/fish.data --sr False --expFolder testfirst	--expID 4',


]

def check_name(cmd):
    if '--multi-scale True' in cmd:
        cmd=cmd.replace('--multi-scale True','--multi-scale')
    if '--multi-scale False' in cmd:
        cmd = cmd.replace('--multi-scale False', '')
    if '--rect True' in cmd:
        cmd=cmd.replace('--rect True','--rect')
    if '--rect False' in cmd:
        cmd=cmd.replace('--rect False','')
    if '--freeze True' in cmd:
        cmd=cmd.replace('--freeze True','--freeze')
    if '--freeze False' in cmd:
        cmd=cmd.replace('--freeze False','')
    if '--sr True' in cmd:
        cmd = cmd.replace('--sr True', '--sr')
    if '--sr False' in cmd:
        cmd = cmd.replace('--sr False', '')
    return cmd

import os
log = open("train_log.log", "a+")
for cmd in cmds:
    log.write(cmd)
    log.write("\n")
    cmd=check_name(cmd)
    print(cmd)
    os.system(cmd)
