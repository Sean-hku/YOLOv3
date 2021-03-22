import argparse
import json
import pandas as pd
from torch.utils.data import DataLoader
import csv
from models import *
from utils.datasets import *
from utils.utils import *
import copy
import box_postprocess as bp
import cv2
import numpy as np

def Vis(path,boxes,black_path,black_box,merge_box):

    img = cv2.imread(path[0])
    img = cv2.resize(img, (416, 416))
    if boxes[0] is not None:
        for box in boxes[0].tolist():
            prob =str(round(box[4],2))
            img = cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,255),2)
            cv2.putText(img,prob,(int(box[0]),int(box[1])),cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
    #black_img
    img_black = cv2.imread(black_path[0])
    img_black = cv2.resize(img_black,(416,416))
    if black_box[0] is not None:
        for b_box in black_box[0].tolist():
            prob =str(round(b_box[4],2))
            img_black = cv2.rectangle(img_black,(int(b_box[0]),int(b_box[1])),(int(b_box[2]),int(b_box[3])),(0,255,255),2)
            cv2.putText(img_black,prob,(int(b_box[0]),int(b_box[1])),cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
    img_merge = cv2.imread(black_path[0])
    img_merge = cv2.resize(img_merge,(416,416))
    if merge_box is not None:
        for m_box in merge_box.tolist():
            prob = str(round(m_box[4], 2))
            img_merge = cv2.rectangle(img_merge, (int(m_box[0]), int(m_box[1])), (int(m_box[2]), int(m_box[3])),
                                      (0, 255, 255), 2)
            cv2.putText(img_merge, prob, (int(m_box[0]), int(m_box[1])), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    all_img = np.hstack([img,img_black,img_merge])
    cv2.imshow("merge", all_img)
    # cv2.imwrite(os.path.join('./result/merge_result', path[0].split('/')[5]), all_img)
    cv2.waitKey(1)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def ensemble_test(cfg,
         black_cfg,
         data,
         black_data_path,
         black_weights=None,
         weights=None,
         batch_size=1,
         img_size=416,
         iou_thres=0.5,
         conf_thres=0.7,
         nms_thres=0.5,):
    # Initialize/load model and set device
    device = ''
    device = torch_utils.select_device(device)
    verbose = True

    # Initialize  model
    model = Darknet(cfg, img_size).to(device)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)


    # Initialize black_model
    black_model = Darknet(black_cfg, img_size).to(device)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        black_model.load_state_dict(torch.load(black_weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(black_model, black_weights)


    # Configure run
    data = parse_data_cfg(data)
    nc = int(data['classes'])  # number of classes
    test_path = data['valid']  # path to test images
    names = load_classes(data['names'])  # class names

    # Dataloader
    dataset = LoadImagesAndLabels(test_path, img_size, batch_size, rect=False)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=1,
                            shuffle=False,
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)
    #black dataloader
    black_data = parse_data_cfg(black_data_path)
    black_test_path = black_data['valid']
    black_dataset = LoadImagesAndLabels(black_test_path, img_size, batch_size, rect=False)
    black_dataloader = DataLoader(black_dataset,
                            batch_size=batch_size,
                            num_workers=1,
                            shuffle=False,
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)

    seen = 0
    model.eval()
    black_model.eval()
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1')
    p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3)
    write_tb = True
    jdict, stats, ap, ap_class = [], [], [], []
    all_path = []
    b_list = list(black_dataloader)
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        # black_targets = black_dataloader[batch_i][1].to(device)
        # print("gray_path",paths)
        black_imgs = b_list[batch_i][0].to(device)
        black_path = b_list[batch_i][2]
        # print("black_path",black_path)
        black_imgs = black_imgs.to(device)
        targets = targets.to(device)
        imgs = imgs.to(device)
        _, _, height, width = imgs.shape  # batch size, channels, height, width

        # Plot images with bounding boxes
        if batch_i == 0 and not os.path.exists('test_batch0.jpg'):
            plot_images(imgs=imgs, targets=targets, paths=paths, fname='test_batch0.jpg')

        # Run model
        inf_out, train_out = model(imgs)  # inference and training outputs
        #black model
        black_inf_out, black_train_out = black_model(black_imgs)
        # Compute loss
        if hasattr(model, 'hyp'):  # if model has loss hyperparameters
            loss += compute_loss(train_out, targets, model)[1][:3].cpu()  # GIoU, obj, cls
        #black model
        if hasattr(black_model, 'hyp'):  # if model has loss hyperparameters
            loss += compute_loss(black_train_out, targets, black_model)[1][:3].cpu()  # GIoU, obj, cls

        # Run NMS
        output = non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres)

        black_output = non_max_suppression(black_inf_out, conf_thres=conf_thres, nms_thres=nms_thres)
        # print("gray",output)
        # print("black",black_output)

        output_list = []
        for i in range(len(output)):
            BE = bp.BoxEnsemble()
            merge_output = BE.ensemble_box(output[i],black_output[i])
            # print("merge:",merge_output)
            output_list.append(merge_output)

        Vis(paths, output, black_path, black_output, merge_output)
        # Statistics per image
        for si, pred in enumerate(output_list):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                continue



            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Assign all predictions as incorrect
            correct = [0] * len(pred)
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                tbox[:, [0, 2]] *= width
                tbox[:, [1, 3]] *= height

                # Search for correct predictions
                for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                    # Break if all targets already located in image
                    if len(detected) == nl:
                        break

                    # Continue if predicted class not among image classes
                    if pcls.item() not in tcls:
                        continue

                    # Best iou, index between pred and targets
                    m = (pcls == tcls_tensor).nonzero().view(-1)
                    iou, bi = bbox_iou(pbox, tbox[m]).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if iou > iou_thres and m[bi] not in detected:  # and pcls == tcls[bi]:
                        correct[i] = 1
                        detected.append(m[bi])

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))

    # Compute statistics
    stats_copy = copy.deepcopy(stats)
    for x in list(zip(*stats_copy))[0]:
        if len(x) == 0:
            x.append(2)
    correct_img = list(list(zip(*stats_copy))[0])
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)


    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))
    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map, mf1, *(loss / len(dataloader)).tolist()), maps


if __name__ == '__main__':
    gray_model = '/media/hkuit164/WD20EJRX/result/best_finetune/gray/SLIM-prune_0.95_keep_0.1/best.weights'
    # gray_model = '/home/user/Documents/yolov3-channel-and-layer-pruning/weights/gray/26/best.weight'
    gray_cfg = '/media/hkuit164/WD20EJRX/result/best_finetune/gray/SLIM-prune_0.95_keep_0.1/prune_0.95_keep_0.1.cfg'
    # gray_cfg = '/home/user/Documents/yolov3-channel-and-layer-pruning/cfg/yolov3-original-1cls-leaky.cfg'
    black_model = '/media/hkuit164/WD20EJRX/result/best_finetune/black/SLIM-prune_0.93_keep_0.1/best.weights'
    black_cfg = '/media/hkuit164/WD20EJRX/result/best_finetune/black/SLIM-prune_0.93_keep_0.1/prune_0.93_keep_0.1.cfg'
    data = '/media/hkuit164/WD20EJRX/yolov3-channel-and-layer-pruning/data/ensemble/gray/all.data'
    black_data = '/media/hkuit164/WD20EJRX/yolov3-channel-and-layer-pruning/data/ensemble/black/all.data'
    with torch.no_grad():
        ensemble_test(gray_cfg,black_cfg,data,black_data,black_model,gray_model)