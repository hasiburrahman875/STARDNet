# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Logging utils
"""

import os
import warnings
from threading import Thread

import pkg_resources as pkg
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.general import colorstr, emojis
from utils.loggers.wandb.wandb_utils import WandbLogger
from utils.plots import plot_images, plot_images_temporal, plot_results
from utils.torch_utils import de_parallel

LOGGERS = ('wandb', 'csv', 'tb')  # text-file, TensorBoard, Weights & Biases
RANK = int(os.getenv('RANK', -1))

# Optional Weights & Biases import with safe login
try:
    import wandb

    assert hasattr(wandb, '__version__')  # verify package import not local dir
    if pkg.parse_version(wandb.__version__) >= pkg.parse_version('0.12.2') and RANK in [0, -1]:
        wandb_login_success = wandb.login(timeout=30)
        if not wandb_login_success:
            wandb = None
except (ImportError, AssertionError):
    wandb = None


class Loggers:
    """YOLOv5 Loggers class"""

    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=None, include=LOGGERS):
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
        self.logger = logger  # for printing results to console
        self.include = include
        self.keys = [
            'train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
            'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',  # metrics
            'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
            'x/lr0', 'x/lr1', 'x/lr2'  # params
        ]
        for k in LOGGERS:
            setattr(self, k, None)  # init empty logger attributes (wandb, csv, tb)
        self.csv = True  # always log to csv

        # Always define tb so later checks don't crash when TB is disabled
        self.tb = None

        # Message about W&B availability
        if not wandb:
            prefix = colorstr('Weights & Biases: ')
            s = f"{prefix}run 'pip install wandb' to automatically track and visualize YOLOv5 🚀 runs (RECOMMENDED)"
            print(emojis(s))

        # TensorBoard
        s = self.save_dir
        if 'tb' in self.include and not getattr(self.opt, 'evolve', False):
            prefix = colorstr('TensorBoard: ')
            if self.logger:
                self.logger.info(f"{prefix}Start with 'tensorboard --logdir {s.parent}', view at http://localhost:6006/")
            self.tb = SummaryWriter(str(s))

        # Weights & Biases
        if wandb and 'wandb' in self.include:
            wandb_artifact_resume = isinstance(getattr(self.opt, 'resume', None), str) and self.opt.resume.startswith('wandb-artifact://')
            run_id = None
            if getattr(self.opt, 'resume', False) and not wandb_artifact_resume:
                try:
                    run_id = torch.load(self.weights).get('wandb_id')
                except Exception:
                    run_id = None
            self.opt.hyp = self.hyp  # add hyperparameters
            self.wandb = WandbLogger(self.opt, run_id)
        else:
            self.wandb = None

    # ----------------------------- Callbacks -----------------------------

    def on_pretrain_routine_end(self):
        # Callback runs on pre-train routine end
        paths = self.save_dir.glob('*labels*.jpg')  # training labels
        if self.wandb:
            self.wandb.log({"Labels": [wandb.Image(str(x), caption=x.name) for x in paths]})

    def on_train_batch_end(self, ni, model, imgs, targets, paths, plots, sync_bn):
        # Callback runs on train batch end
        if plots:
            # unwrap DataParallel/DistributedDataParallel so we can access custom attrs on the real module
            m = de_parallel(model)
            num_frames = getattr(m, 'num_frames', 1)

            if ni == 0:
                # tb.add_graph only if TB writer exists and we're not using sync BN
                if self.tb and not sync_bn:  # known issue with --sync-bn
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')  # suppress JIT trace warnings
                        try:
                            self.tb.add_graph(torch.jit.trace(m, imgs[0:num_frames], strict=False), [])
                        except Exception:
                            # Avoid crashing training if graph tracing fails
                            pass

            if ni < 3:
                f = self.save_dir / f'train_batch{ni}.jpg'  # filename
                # plot_images supports passing num_frames for temporal mosaics
                Thread(target=plot_images, args=(imgs, targets, paths, f, num_frames), daemon=True).start()

            if self.wandb and ni == 10:
                files = sorted(self.save_dir.glob('train*.jpg'))
                self.wandb.log({'Mosaics': [wandb.Image(str(f), caption=f.name) for f in files if f.exists()]})

    def on_train_epoch_end(self, epoch):
        # Callback runs on train epoch end
        if self.wandb:
            self.wandb.current_epoch = epoch + 1

    def on_val_image_end(self, pred, predn, path, names, im):
        # Callback runs on val image end
        if self.wandb:
            self.wandb.val_one_image(pred, predn, path, names, im)

    def on_val_end(self):
        # Callback runs on val end
        if self.wandb:
            files = sorted(self.save_dir.glob('val*.jpg'))
            self.wandb.log({"Validation": [wandb.Image(str(f), caption=f.name) for f in files]})

    def on_fit_epoch_end(self, vals, epoch, best_fitness, fi):
        # Callback runs at the end of each fit (train+val) epoch
        x = {k: v for k, v in zip(self.keys, vals)}  # dict

        # CSV logging
        if self.csv:
            file = self.save_dir / 'results.csv'
            n = len(x) + 1  # number of cols
            header = (('%20s,' * n) % tuple(['epoch'] + self.keys)).rstrip(',') + '\n'
            line = ('%20.5g,' * n) % tuple([epoch] + vals)
            with open(file, 'a') as f:
                if not file.exists() or file.stat().st_size == 0:
                    f.write(header)
                f.write(line.rstrip(',') + '\n')

        # TensorBoard scalars
        if self.tb:
            for k, v in x.items():
                self.tb.add_scalar(k, v, epoch)

        # Weights & Biases scalars
        if self.wandb:
            self.wandb.log(x)
            self.wandb.end_epoch(best_result=best_fitness == fi)

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        # Callback runs on model save event
        if self.wandb:
            save_period = getattr(self.opt, 'save_period', -1)
            if ((epoch + 1) % save_period == 0 and not final_epoch) and save_period != -1:
                self.wandb.log_model(last.parent, self.opt, epoch, fi, best_model=best_fitness == fi)

    def on_train_end(self, last, best, plots, epoch, results):
        # Callback runs on training end
        if plots:
            plot_results(file=self.save_dir / 'results.csv')  # save results.png

        files = ['results.png', 'confusion_matrix.png', *(f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R'))]
        files = [(self.save_dir / f) for f in files if (self.save_dir / f).exists()]  # filter

        # TensorBoard images
        if self.tb:
            import cv2
            for f in files:
                try:
                    self.tb.add_image(f.stem, cv2.imread(str(f))[..., ::-1], epoch, dataformats='HWC')
                except Exception:
                    pass

        # Weights & Biases images and model artifact
        if self.wandb:
            self.wandb.log({"Results": [wandb.Image(str(f), caption=f.name) for f in files]})
            # Calling wandb.log. TODO: Refactor this into WandbLogger.log_model
            if not getattr(self.opt, 'evolve', False):
                try:
                    wandb.log_artifact(
                        str(best if best.exists() else last),
                        type='model',
                        name='run_' + self.wandb.wandb_run.id + '_model',
                        aliases=['latest', 'best', 'stripped']
                    )
                except Exception:
                    pass
                self.wandb.finish_run()
            else:
                self.wandb.finish_run()
                self.wandb = WandbLogger(self.opt)
