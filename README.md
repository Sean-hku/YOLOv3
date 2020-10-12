# Pytorch-YOLOv3<br>
## Baseline training<br>
**Parse csv files:** `python csv2cmd.py` then you will get cmd.txt<br>
**Assign cmds to GPU:** `python customized_cmd.py` then you will get training cmds<br>
**start training:** `python train.py --cfg --weights`

## Sparse training <br>
**strat sparse training** `python train_sparse.py --cfg cfg/my_cfg.cfg --data data/my_data.data --weights weights/last.weights --epochs 300 --batch-size 32 -sr --s 0.001 --prune 1`<br>
## Test sparse ratio
`python auto_sparse.py` you will get a csv about bn_weight and sparse rate
## Pruning
**two methods** **Slim_prune** `prune/slim_prune.py` **Layer_channel_prune** `prune/layer_channel_prune.py` or you can also run `auto_pruning.py` if you want both pruning methods.<br>
Before pruning, you should edit cdf/config.py <br>
```python
models = {
"weights/sparse/gray26_sE-3/last.pt": "cfg/yolov3-original-1cls-leaky.cfg",
}
slim_params = [(0.95, 0.1)] 0.95:global_percent 0.1: layer_keep
all_prune_params = [(10, 0.95, 0.01), (15, 0.95, 0.01)] 10/15:shortcut layer num 0.95:global_percent 0.1: layer_keep
