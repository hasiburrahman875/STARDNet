# STARD-Net

**STARD-Net: SpatioTemporal Attention for Robust Detection of Tiny Airborne Objects from Moving Drones**  
*ACM Transactions on Spatial Algorithms and Systems (TSAS), Volume 12, Issue 1, Article 3, 2026*

[![Paper](https://img.shields.io/badge/Paper-ACM-blue)](https://doi.org/10.1145/3787467)
[![Dataset](https://img.shields.io/badge/Dataset-Download-green)](https://mailmissouri-my.sharepoint.com/:f:/g/personal/mrpk9_umsystem_edu/EtkiYMp_l7pKhGL_yuSdfPMBiLBS1wxl3w2xRRcO3Es7Fw?e=I7Gs0e)
[![Weights](https://img.shields.io/badge/Weights-Download-orange)](https://mailmissouri-my.sharepoint.com/:f:/g/personal/mrpk9_umsystem_edu/IgDsAlG-ulgFS7NoivRHxEMwAQGuXz6ChWAsQTALdfFFZCo?e=9RYThm)

Official implementation of **STARD-Net**, an end-to-end framework for detecting tiny airborne objects from moving drones under challenging conditions such as small target size, background clutter, camouflage, and occlusion.

## Paper

**Hasibur Rahman and Sanjay Madria.**  
*STARD-Net: SpatioTemporal Attention for Robust Detection of Tiny Airborne Objects from Moving Drones.*  
ACM Transactions on Spatial Algorithms and Systems, 12(1), Article 3, 2026.  
**DOI:** [10.1145/3787467](https://doi.org/10.1145/3787467)

## Method Overview

STARD-Net combines spatial and temporal modeling for airborne object detection from moving-camera video. The main components are:

- **Drone Receptive Field Block (DRFB):** extracts multiscale and geometrically diverse features for small and visually difficult airborne targets.
- **ConvLSTM:** models motion and temporal dependencies across consecutive frames while preserving spatial structure.
- **SpatioTemporal Attention Block (STAB):** aggregates spatial and temporal context to strengthen features when appearance cues are weak or partially occluded.
- **CSP/Swin-based detection heads:** introduce global context while retaining an efficient multi-scale detection structure.

The repository provides the model implementation, dataset configuration files, training code, validation code, and supporting scripts used for STARD-Net experiments.

## Resources

- **Paper:** [ACM Digital Library / DOI](https://doi.org/10.1145/3787467)
- **Datasets:** [Download](https://mailmissouri-my.sharepoint.com/:f:/g/personal/mrpk9_umsystem_edu/EtkiYMp_l7pKhGL_yuSdfPMBiLBS1wxl3w2xRRcO3Es7Fw?e=I7Gs0e)
- **Pretrained weights:** [Download](https://mailmissouri-my.sharepoint.com/:f:/g/personal/mrpk9_umsystem_edu/IgDsAlG-ulgFS7NoivRHxEMwAQGuXz6ChWAsQTALdfFFZCo?e=9RYThm)

## Repository Structure

Important files and directories include:

```text
STARDNet/
├── models/
│   └── stardnet.yaml
├── data/
│   ├── mvaaod.yaml
│   ├── mvaaod-s.yaml
│   ├── FLDrone.yaml
│   ├── NPS.yaml
│   └── hyps/
├── train.py
├── val.py
├── detect.py
├── inference.py
├── requirements.txt
└── conversion_scripts/
```

## Installation

Clone the repository:

```bash
git clone https://github.com/hasiburrahman875/STARDNet.git
cd STARDNet
```

Create a Python environment and install the dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```bash
venv\Scripts\activate
```

## Dataset Preparation

Download the required datasets from the link above and update the corresponding YAML configuration with the local paths to the training, validation, testing, and video data.

For example:

```text
data/mvaaod.yaml
data/FLDrone.yaml
data/NPS.yaml
```

The YAML file should contain the correct dataset paths, number of classes, and class names for the experiment being run.

### FL-Drone

If you encounter issues preparing the **FL-Drone** dataset, please refer to the dataset preparation instructions provided with [TransVisDrone](https://github.com/tusharsangam/TransVisDrone).

## Validation

Download the pretrained weights and run validation with the appropriate dataset configuration:

```bash
python val.py \
  --data data/dataset_name.yaml \
  --weights path/to/weights.pt \
  --img 1280 \
  --batch 1 \
  --num-frame 5 \
  --name experiment_name
```

Replace `dataset_name.yaml`, the weight path, and image size as required for the dataset being evaluated.

## Training

A typical STARD-Net training command is:

```bash
python train.py \
  --data data/dataset_name.yaml \
  --hyp data/hyps/hyp.UAVDT.yaml \
  --img 1280 \
  --device 0,1 \
  --batch 2 \
  --cfg models/stardnet.yaml \
  --epoch 300 \
  --adam \
  --num-frame 5 \
  --name experiment_name
```

Adjust the dataset configuration, GPU devices, batch size, image size, and other parameters according to the available hardware and experiment setup.

## Citation

If you use STARD-Net or this code in your research, please cite:

```bibtex
@article{rahman2026stardnet,
  author  = {Rahman, Hasibur and Madria, Sanjay},
  title   = {STARD-Net: SpatioTemporal Attention for Robust Detection of Tiny Airborne Objects from Moving Drones},
  journal = {ACM Transactions on Spatial Algorithms and Systems},
  volume  = {12},
  number  = {1},
  articleno = {3},
  year    = {2026},
  doi     = {10.1145/3787467}
}
```

## Contact

For questions about the implementation or experiments, please open an issue in this repository.
