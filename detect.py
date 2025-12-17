# YOLOv5 by Ultralytics, GPL 3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset
"""

import argparse
import json, pickle
import os
import sys
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.experimental import attempt_load
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.general import (
    LOGGER, box_iou, check_dataset, check_img_size, check_requirements, check_suffix, check_yaml,
    coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
    scale_coords, xywh2xyxy, xyxy2xywh
)
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.plots import output_to_target, plot_images, Annotator, colors
from utils.torch_utils import select_device, time_sync


def save_one_txt(predn, save_conf, shape, file):
    gn = torch.tensor(shape)[[1, 0, 1, 0]]
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


aot_results = []
def save_aot_one_pkl(predn, path, file_path=False):
    if not file_path:
        track_id = 0
        for *xyxy, conf, cls in predn.tolist():
            if int(cls) == 0:
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                result = {
                    "detections": [
                        {
                            "track_id": track_id,
                            "x": float(xywh[0]),
                            "y": float(xywh[1]),
                            "w": float(xywh[2]),
                            "h": float(xywh[3]),
                            "n": "airborne",
                            "s": float(conf)
                        }
                    ],
                    "img_name": os.path.basename(path)
                }
                track_id += 1
                aot_results.append(result)
    else:
        os.makedirs(str(Path(file_path).parent), exist_ok=True)
        pickle.dump(aot_results, open(file_path, "wb"))


def save_one_json(predn, jdict, path, class_map):
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])
    box[:, :2] -= box[:, 2:] / 2
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)
        })


jdict_gt = {}
def save_one_json_with_gt(predn, path, class_map, labelsn):
    if len(predn) > 0 or len(labelsn) > 0:
        image_id = int(path.stem) if path.stem.isnumeric() else path.stem
        box = predn[:, :4]
        jdict_gt[image_id] = {"detections": [], "labels": []}
        for p, b in zip(predn.tolist(), box.tolist()):
            jdict_gt[image_id]["detections"].append({
                'bbox': [round(x, 3) for x in b],
                'score': round(p[4], 5),
                'category_id': class_map[int(p[5])]
            })
        box = labelsn[:, 1:]
        for p, b in zip(labelsn[:, 0], box.tolist()):
            jdict_gt[image_id]["labels"].append({'bbox': [round(x, 3) for x in b], 'category_id': int(p)})


def save_one_json_with_gt_dump(file_path):
    os.makedirs(str(Path(file_path).parent), exist_ok=True)
    pickle.dump(jdict_gt, open(file_path, "wb"))
    print("Predictions with gt dumped")


def get_data_split_number(data_yaml_file_path):
    split_number = os.path.basename(data_yaml_file_path).split(".")[0].split("_")[-1]
    return int(split_number) if split_number.isnumeric() else 0


def process_batch(detections, labels, iouv):
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


def is_fourth_frame(path):
    return True if int(os.path.basename(path).split(".")[0].split("_")[-1]) % 4 == 0 else False


@torch.no_grad()
def run(
    data,
    weights=None,
    batch_size=32,
    imgsz=640,
    conf_thres=0.001,
    iou_thres=0.5,
    task='val',
    device='',
    single_cls=False,
    augment=False,
    verbose=False,
    save_txt=False,
    save_hybrid=False,
    save_conf=False,
    save_json=False,
    save_json_gt=False,
    project=ROOT / 'runs/val',
    name='exp',
    exist_ok=False,
    half=False,
    num_frames=5,
    every_fourth_frame=False,
    save_aot_predictions=False,
    model=None,
    dataloader=None,
    save_dir=Path(''),
    plots=True,
    callbacks=Callbacks(),
    compute_loss=None,
    save_img=False,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False
):
    epoch_number = -1
    training = model is not None
    if training:
        device = next(model.parameters()).device
    else:
        device = select_device(device, batch_size=batch_size)
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
        if save_img:
            (save_dir / 'images').mkdir(parents=True, exist_ok=True)

        check_suffix(weights, '.pt')
        model, epoch_number = attempt_load(weights, map_location=device, return_epoch_number=True)
        gs = max(int(model.stride.max()), 32)
        imgsz = check_img_size(imgsz, s=gs)

        data_yaml_path = None
        if isinstance(data, (str, Path)):
            data_yaml_path = data
        if data_yaml_path is None:
            assert save_aot_predictions == False, print("please launch with --data when saving AOT predictions")
        data = check_dataset(data)

    half &= device.type != 'cpu'
    model.half() if half else model.float()

    model.eval()
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')
    nc = 1 if single_cls else int(data['nc'])
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()

    if not training:
        if device.type != 'cpu':
            model(torch.zeros(num_frames, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
        pad = 0.0 if task == 'speed' else 0.5
        task = task if task in ('train', 'val', 'test') else 'val'
        annotation_test_path = data.get(f"annotation_{task}", "")
        video_root = data.get(f"video_root_path_{task}", "")
        dataloader = create_dataloader(
            data[task], annotation_test_path, video_root, imgsz, batch_size, gs, single_cls,
            pad=pad, rect=True, prefix=colorstr(f'{task}: '), is_training=False, num_frames=num_frames
        )[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 7) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'epoch(' + str(epoch_number) + ')')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict = []
    stats, ap, ap_class = [], [], []

    for batch_i, (img, targets, paths, shapes, main_target_indices, label_paths) in enumerate(tqdm(dataloader, desc=s)):
        t1 = time_sync()
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()
        img /= 255
        targets = targets.to(device)

        nb, _, height, width = img.shape
        t2 = time_sync()
        dt[0] += t2 - t1

        out, train_out = model(img, augment=augment)
        dt[1] += time_sync() - t2

        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]

        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []
        t3 = time_sync()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=False, agnostic=single_cls)
        dt[2] += time_sync() - t3

        for si, pred in enumerate(out):
            if si not in main_target_indices:
                continue
            if every_fourth_frame and not is_fourth_frame(paths[si]):
                continue

            labels = targets[targets[:, 0] == si, 1:]

            tbox = xywh2xyxy(labels[:, 1:5])
            scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)
            labelsn = [labelsni for labelsni in labelsn if labelsni[1:].any()]
            labelsn = torch.cat(labelsn, 0).reshape(-1, 5) if len(labelsn) > 0 else torch.zeros((0, 5)).to(device)
            nl = len(labelsn)

            tcls = labelsn[:, 0].tolist() if nl else []
            path = Path(paths[si])
            shape = shapes[si][0]

            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])

            if nl:
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

            # txt save like detect.py
            if save_txt and len(predn):
                txt_path = str(save_dir / 'labels' / path.stem) + '.txt'
                gn = torch.tensor(shape)[[1, 0, 1, 0]]
                with open(txt_path, 'a') as f:
                    for *xyxy, conf, cls in predn.tolist():
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # image save like detect.py
            if save_img:
                im0 = cv2.imread(str(path))
                if im0 is None:
                    h, w = int(shape[0]), int(shape[1])
                    im0 = np.zeros((h, w, 3), dtype=np.uint8)
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                for *xyxy, conf, cls in predn.tolist():
                    c = int(cls)
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                im0 = annotator.result()
                save_path = str(save_dir / 'images' / path.name)
                cv2.imwrite(save_path, im0)

            if save_json:
                save_one_json(predn, jdict, path, class_map)
            if save_aot_predictions:
                save_aot_one_pkl(predn, path, False)
            if save_json_gt:
                save_one_json_with_gt(predn, path, class_map, labelsn)
            callbacks.run('on_val_image_end', pred, predn, path, names, img[si])

        if plots and batch_i < 3:
            f = save_dir / f'val_batch{batch_i}_labels.jpg'
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'val_batch{batch_i}_pred.jpg'
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    stats = [np.concatenate(x, 0) for x in zip(*stats)] if len(stats) else []
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)
    else:
        nt = torch.zeros(1)

    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4 + '%11i'
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map, epoch_number))

    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i], epoch_number))

    t = tuple(x / seen * 1E3 for x in dt) if seen else (0.0, 0.0, 0.0)
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')

    if not training:
        with open(str(save_dir / "results.txt"), 'w') as f:
            pf = '%20s' + '%11i' * 2 + '%11.3g' * 4 + '%11i'
            f.writelines([s, "\n", pf % ('all', seen, nt.sum(), mp, mr, map50, map, epoch_number)])

    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')
        pred_json = str(save_dir / f"{w}_predictions.pkl")
        LOGGER.info(f'\nEvaluating pycocotools mAP, saving {pred_json}')
        with open(pred_json, 'wb') as f:
            pickle.dump(jdict, f)
        try:
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            anno = COCO(anno_json)
            pred = anno.loadRes(pred_json)
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    if save_aot_predictions:
        save_aot_one_pkl(False, False, save_dir / 'aotpredictions' / ('predictions_split_' + str(get_data_split_number(data_yaml_path)) + '.pkl'))
    if save_json_gt:
        save_one_json_with_gt_dump(save_dir / 'predictionsgt' / ('predictionsgt_split_' + str(get_data_split_number(data_yaml_path)) + '.pkl'))

    model.float()
    if not training:
        s_msg = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s_msg}")

    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class) if len(stats) and stats[0].any() else []:
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model pt path')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size in pixels')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device like 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label plus prediction hybrid results to txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in saved txt')
    parser.add_argument('--save-json', action='store_true', help='save a COCO JSON results file')
    parser.add_argument('--save-json-gt', action='store_true', help='save predictions with gt into a pickle')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project or name')
    parser.add_argument('--name', default='exp', help='save to project or name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project or name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 inference')
    parser.add_argument('--num-frames', type=int, default=5, help='number of frames to load')
    parser.add_argument('--every-fourth-frame', action='store_true', help='record results on every fourth frame')
    parser.add_argument('--save-aot-predictions', action='store_true', help='store predictions in AOT style')
    parser.add_argument('--save-img', action='store_true', help='save rendered images with detections')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness in pixels')
    parser.add_argument('--hide-labels', action='store_true', help='hide labels on saved images')
    parser.add_argument('--hide-conf', action='store_true', help='hide confidences on saved images')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)
    opt.save_json |= str(opt.data).endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    if opt.task in ('train', 'val', 'test'):
        run(**vars(opt))
    elif opt.task == 'speed':
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=opt.imgsz, conf_thres=opt.conf_thres,
                iou_thres=opt.iou_thres, device=opt.device, save_json=False, plots=False)
    elif opt.task == 'study':
        x = list(range(256, 1536 + 128, 128))
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'
            y = []
            for i in x:
                LOGGER.info(f'\nRunning {f} point {i}...')
                r, _, t = run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=i, conf_thres=opt.conf_thres,
                              iou_thres=opt.iou_thres, device=opt.device, save_json=opt.save_json, plots=False)
                y.append(r + t)
            np.savetxt(f, y, fmt='%10.4g')
        plot_val_study(x=x)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
