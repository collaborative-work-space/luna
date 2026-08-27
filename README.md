<h1 align="center">LUNA: Low-Light Robust Panoptic Lifting for Adverse Robotic 3D Scene Perception</h1>

<p align="center">
  <a href="https://people.csiro.au/r/a/ahalya-ravendran">Ahalya Ravendran</a>, 
  <a href="https://people.csiro.au/L/X/Xun-Li">Xun Li</a>
  <a href="https://www.qut.edu.au/about/our-people/academic-profiles/leo.lebrat">Leo Lebrat</a>, 
  <a href="https://www.qut.edu.au/about/our-people/academic-profiles/rodrigo.santacruz">Rodrigo Santa Cruz</a>, 
  <a href="https://people.csiro.au/z/h/hu1-zhang">Hu Zhang</a>, 
  <a href="https://people.csiro.au/P/L/Lars-Petersson">Lars Petersson</a>, 
  <a href="https://people.csiro.au/W/D/Dadong-Wang">Dadong Wang</a>, 
</p>

<p align="center">
  CSIRO and Queensland University of Technology*, Australia
</p>

<p align="center">
  <a href="https://collaborative-work-space.github.io/luna/">Project Page</a>
</p>


## 📌 Overview
**LUNA** is a geometry-aware panoptic lifting framework for robust 3D scene perception under low-light, noisy, and motion-blurred imaging conditions. It extends [Panoptic Lifting](https://github.com/nihalsid/panoptic-lifting) with geometry-aware depth
supervision and [Fast Adaptive Multitask Optimization (FAMO)](https://github.com/Cranial-XIX/FAMO) to keep 3D reconstruction and panoptic segmentation stable when RGB cues alone become unreliable. On a systematically degraded version of the Replica dataset (noise + motion blur, three severity levels each), LUNA consistently outperforms Panoptic Lifting and restoration-augmented baselines in both reconstruction quality (PSNR/SSIM) and panoptic accuracy (mIoU/PQ/SQ/RQ).

## ⚙️ Environment Setup

### 1. Create the environment
```bash
conda create -n luna python=3.9 -y
conda activate luna
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Install LUNA's dependencies
```bash
pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-<your-torch-version>+<your-cuda>.html
```

### 3. Prepare data
LUNA trains on [Replica](https://github.com/facebookresearch/Replica-Dataset) scenes processed the same way as Panoptic Lifting: `src/dataset/preprocessing/preprocess_replica.py` generates Mask2Former semantic/instance pseudo-labels and depth.

### 4. Generate the degraded evaluation variants
```bash
# Gaussian noise: sigma = 40 / 80 / 120 for Easy / Moderate / Strong
python src/util/add-noise.py --input-dir data/ourreplica/<scene>/original/color \
    --output-dir data/ourreplica/<scene>/noise-120/color --sigma 120

# Motion blur: kernel = 10 / 20 / 30 px for Easy / Moderate / Strong
python src/util/add-blur.py --input-dir data/ourreplica/<scene>/original/color \
    --output-dir data/ourreplica/<scene>/blur-30/color --kernel-size 30
```
See `src/dataset/README.md` for which dataset loader variant pairs with each degraded export.

### 5. Verify installation
```bash
cd src
python trainer/train_panopli_tensorf.py \
  experiment=smoke_test dataset_root=<path-to-a-prepared-scene> \
  max_epochs=1 trainer.devices=1 trainer.strategy=auto
```
You should see `train/loss_rgb`, `train/loss_depth`, `train/loss_semantics`, and `famo/weight_*` logged within the first few steps.

## 🚀 Training
```bash
cd src
python trainer/train_panopli_tensorf.py \
  --config-dir config --config-name panopli \
  experiment=<run-name> dataset_root=<path-to-prepared-scene> \
  trainer.accelerator=gpu trainer.devices=4 trainer.strategy=ddp
```

## 📊 Evaluation
```bash
cd src
# Render novel views + panoptic predictions from a checkpoint
python inference/render_panopli.py runs/<run-dir>/checkpoints/<ckpt>.ckpt True

# Compute PSNR / SSIM / mIoU / PQ against ground truth
python inference/evaluate.py --root_path <path-to-original-scene> --exp_path runs/<run-dir>
```

## 📂 Repository Structure

```text
luna/
│── src/                  # Training, inference, dataset, model, and FAMO integration code
│── images/               # Project images and figures for the website
│── css/                  # Stylesheets for project website
│── js/                   # Scripts for project website
│── index.html            # Project website main page
│── requirements.txt
│── THIRD_PARTY_NOTICES.md  # License attribution for vendored FAMO code
└── README.md             # This file
```

## 🫡 Acknowledgements
LUNA builds on [Panoptic Lifting](https://github.com/nihalsid/panoptic-lifting) (Siddiqui et al., CVPR 2023) and integrates [FAMO](https://github.com/Cranial-XIX/FAMO) (Liu et al., NeurIPS 2023).

## 📖 Citation
If you found this code/work to be useful in your own research, please consider citing the following:
```bibtex
@inproceedings{ravendran2026luna,
  title={LUNA: Low-Light Robust Panoptic Lifting for Adverse Robotic 3D Scene Perception},
  author={Ravendran, Ahalya and Li, Xun and Lebrat, Leo and Santa Cruz, Rodrigo and Zhang, Hu and Petersson, Lars and Wang, Dadong},
  booktitle={IEEE World Congress on Computational Intelligence (WCCI)},
  year={2026}
}
```
