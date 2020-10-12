# Pytorch-YOLOv3<br>
## Baseline training<br>
**Parse csv files:** `python csv2cmd.py` then you will get cmd.txt<br>
**Assign cmds to GPU:** `python customized_cmd.py` then you will get training cmds<br>
**Start training:** `python train.py --cfg --weights`
## Baseline testing<br>
**Auto_testing** `python auto_testing.py` you need a csv file and weight folder
## Sparse training <br>
**strat sparse training** `python train_sparse.py --cfg cfg/my_cfg.cfg --data data/my_data.data --weights weights/last.weights --epochs 300 --batch-size 32 -sr --s 0.001 --prune 1`<br>
## Test sparse ratio
`python auto_sparse.py` you will get a csv about bn_weight and sparse rate<br>
## Pruning
**Two methods** **1. Slim_prune** `prune/slim_prune.py` **2. Layer_channel_prune** `prune/layer_channel_prune.py` or you can also run `auto_pruning.py` if you want both pruning methods.<br>
Before pruning, you should modify cfg/config.py <br>
```python
models = {
"weights/sparse/gray26_sE-3/last.pt": "cfg/yolov3-original-1cls-leaky.cfg",
}
slim_params = [(0.95, 0.1)] 0.95:global_percent 0.1: layer_keep
all_prune_params = [(10, 0.95, 0.01), (15, 0.95, 0.01)] 10/15:shortcut layer num 0.95:global_percent 0.1: layer_keep
```
## Finetune
`python auto_finetune.py`<br>
 Before finetune, you should modify cfg/config.py <br>
```
finetune_folders = [
    os.path.join('prune_result/gray26_s{}E-4-last'.format(j), i)
    for j in range(6,10)
    for i in os.listdir('prune_result/gray26_s{}E-4-last'.format(j))
    if os.path.isdir(os.path.join('prune_result/gray26_s{}E-4-last'.format(j), i))
]
```
    Finetune result will be add to prune.csv in corresponding files
