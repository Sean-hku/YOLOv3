import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=273)  # 500200 batches at bs 16, 117263 images = 273 epochs
parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
parser.add_argument('--weight_dir', type=str, default="test")
parser.add_argument('--accumulate', type=int, default=2, help='batches to accumulate before optimizing')
parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp-1cls-leaky.cfg', help='cfg file path')
parser.add_argument('--t_cfg', type=str, default='', help='teacher model cfg file path for knowledge distillation')
parser.add_argument('--data', type=str, default='data/coco.data', help='*.data file path')
parser.add_argument('--test_data', type=str, default='data/ceiling.data', help='*.data file path')
parser.add_argument('--multi-scale', action='store_true', help='adjust (67% - 150%) img_size every 10 batches')
parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
parser.add_argument('--rect', action='store_true', help='rectangular training')
parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
parser.add_argument('--transfer', action='store_true', help='transfer learning')
parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
parser.add_argument('--notest', action='store_true', help='only test final epoch')
parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
parser.add_argument('--img-weights', action='store_true', help='select training images by weight')
parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
parser.add_argument('--weights', type=str, default='', help='initial weights')  # i.e. weights/darknet.53.conv.74
parser.add_argument('--t_weights', type=str, default='', help='teacher model weights')
parser.add_argument('--arc', type=str, default='default', help='yolo architecture')  # defaultpw, uCE, uBCE
parser.add_argument('--prebias', action='store_true', help='transfer-learn yolo biases prior to training')
parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
parser.add_argument('--adam', action='store_true', help='use adam optimizer')
parser.add_argument('--var', type=float, help='debug variable')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.001, help='scale sparse rate')
parser.add_argument('--prune', type=int, default=1, help='0:nomal prune 1:other prune ')
parser.add_argument('--freeze', action='store_true', help='freeze layers ')
parser.add_argument('--expID', type=str, default='0', help='model number')
parser.add_argument('--LR', type=float, default=0.001, help='learning rate')
parser.add_argument('--type', type=str, default='spp', help='yolo type(spp,normal,tiny)')
parser.add_argument('--activation', type=str, default='leaky', help='activation function(leaky,swish,mish)')
parser.add_argument('--expFolder', type=str, default='gray', help='expFloder')
parser.add_argument('--save_interval', default=1, type=int, help='interval')
parser.add_argument('--optimize', type=str, default='sgd', help='optimizer(adam,sgd)')
parser.add_argument('--lr_decay_time', type=int, default=2, help='lr decay time')
parser.add_argument('--write_csv', action='store_true', help='use adam optimizer')
opt = parser.parse_args()

print(opt)