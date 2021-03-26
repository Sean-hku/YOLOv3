#-*-coding:utf-8-*-

cmds = [

'python train.py --type original --activation leaky --batch-size 8 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --s 0.01 --expFolder rgb	--expID 1',
'python train.py --type original --activation leaky --batch-size 8 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --s 0.05 --expFolder rgb	--expID 2',
'python train.py --type original --activation leaky --batch-size 8 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize adam --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --s 0.005 --expFolder rgb	--expID 3',


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
