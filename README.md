# ContourFD-Net

**ContourFD-Net: Finite Difference Gradient-Guided Contour-Aware Attention Network for Medical Image Segmentation**

ContourFD-Net is a contour-aware medical image segmentation network that leverages finite-difference gradient guidance and attention mechanisms to better capture lesion boundaries, especially in challenging medical images such as breast ultrasound.

<p align="center">
    <img src="./assets/fives_viz.png" width="800" />
</p>

---

## 🔗 Pretrained Weights

We provide pretrained weights to help reproduce the results in the paper.


* **Pre-trained models (four) – ContourFD-Net (best IoU checkpoint)**
  👉 [Download (Google Drive)]([https://your-google-drive-link.com](https://drive.google.com/drive/folders/1nosiIXYoeIE-2ZM8z3vHd627TvxK8cN9?usp=sharing])
  
### How to Use Pretrained Weights

1. 下载 `.pt` 模型文件（例如 `weight_best_iou.pt`）。
2. 建议目录结构如下（也可以用你自己的路径，只要命令行参数一致即可）：

```text
working/checkpoints/ContourFD-Net/20241120-223413/
    cfg.json
    weight_best_iou.pt
```

3. 评估时：

```bash
python eval.py \
  -cfg working/checkpoints/ContourFD-Net/20241120-223413/cfg.json \
  --ckpt working/checkpoints/ContourFD-Net/20241120-223413/weight_best_iou.pt
```

4. 推理时：

```bash
python infer.py \
  -cfg working/checkpoints/ContourFD-Net/20241120-223413/cfg.json \
  --ckpt working/checkpoints/ContourFD-Net/20241120-223413/weight_best_iou.pt \
  --input_dir path/to/your/images \
  --output_dir path/to/save/results
```

---

## 📦 Dependencies

* **Environment**

  * OS: Debian 12 (bookworm)
  * GPU: NVIDIA RTX 3090 / RTX 3080 Ti
  * CUDA: 11.8
  * cuDNN: 8.9.7
  * Python: 3.10.16
  * PyTorch: 2.6.0

---

## 🚀 Installation

1. **Clone this repository**

```bash
git clone https://github.com/ZBKim/ContourFD-Net.git
cd ContourFD-Net
```

2. **Create a conda environment & install dependencies**

**Option A: Manually create environment**

```bash
conda create -n ContourFD python=3.10.16 -y
conda activate ContourFD

# Install PyTorch (CUDA 11.8)
conda install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu118

# Install Python dependencies
pip install -r requirements.txt
```

**Option B: Use the provided environment file**

```bash
conda env create -f environment.yml
conda activate ContourFD
```

---

## 📂 Dataset Preparation

### Breast Ultrasound Images Dataset (BUSI)

* Dataset: **BUSI**

  * Official page: [BUSI Dataset](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)
  * Backup: [Kaggle – Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

1. **Download & extract the dataset**, then organize it as:

```text
working/dataset/BUSI/          # or any path you want
    benign/
        *.png
    malignant/
        *.png
    normal/
        *.png
```

> ⚠️ **Important**: Do NOT change the original filenames of the images.

2. **Combine multiple masks of the same sample**

Some BUSI samples have more than one label file. Merge them using:

```bash
cd src/tools
python busi_01_combine_masks.py --data_root ../working/dataset/BUSI/
```

3. **Split BUSI into 5 folds**

```bash
cd src/tools
python busi_02_split_folds.py --data_root ../working/dataset/BUSI/
```

* The images are **sorted by filename** before splitting.
* Splitting is deterministic: the same fold index will always contain the same samples between runs.
* After this step, the directory will look like:

```text
working/dataset/BUSI/
    benign/
    malignant/
    normal/
    folds/
        fold0/
        fold1/
        fold2/
        fold3/
        fold4/
```

---

## 🏋️ Training

Make sure you are in the project root (e.g. `ContourFD-Net/`) and the dataset path in the config is correct.

### Single-GPU training on BUSI

```bash
python train.py -cfg configs/BUSI.py
```

### Multi-GPU training

```bash
python GPU.py -cfg configs/BUSI.py
```

> 💡 You can customize the config (number of folds, loss functions, batch size, learning rate, etc.) in `configs/BUSI.py`.

---

## 📊 Evaluation

After training, checkpoints and config files are saved under:

```text
working/checkpoints/ContourFD-Net/<timestamp>/
    cfg.json
    weight_latest.pt
    weight_best_iou.pt
    ...
```

### Evaluate with all checkpoints of an experiment

```bash
python eval.py \
  -cfg working/checkpoints/ContourFD-Net/20241120-223413/cfg.json
```

* This will iterate over all available `.pt` files in the corresponding directory and report metrics.

### Evaluate with a specific checkpoint

```bash
python eval.py \
  -cfg working/checkpoints/ContourFD-Net/20241120-223413/cfg.json \
  --ckpt working/checkpoints/ContourFD-Net/20241120-223413/weight_best_iou.pt
```

---

## 🔍 Inference

You can run inference on a folder of images using a trained or pretrained checkpoint.

### Inference with all checkpoints

```bash
python infer.py \
  -cfg working/checkpoints/ContourFD-Net/20241120-223413/cfg.json \
  --input_dir path/to/your/images \
  --output_dir path/to/save/results
```

### Inference with a specific checkpoint

```bash
python infer.py \
  -cfg working/checkpoints/ContourFD-Net/20241120-223413/cfg.json \
  --ckpt working/checkpoints/ContourFD-Net/20241120-223413/weight_best_iou.pt \
  --input_dir path/to/your/images \
  --output_dir path/to/save/results
```

* `--input_dir` 支持一个包含待分割图像的目录。
* `--output_dir` 将保存模型输出的分割结果（通常为 mask / overlay）。

---

## 📌 Notes

* Please make sure the paths in the config file (`configs/BUSI.py` or your custom config) match your dataset and working directories.
* For reproducibility, you can fix random seeds in the config or training script (PyTorch / NumPy / Python `random`).

---

## 📝 Citation

If you find this repository useful in your research, please consider citing:

```bibtex
@article{your_contourfdnet_paper,
  title   = {ContourFD-Net: Finite Difference Gradient-Guided Contour-Aware Attention Network for Medical Image Segmentation},
  author  = {Your Name and Others},
  journal = {Journal Name},
  year    = {2024}
}
```

(请根据你的真实论文信息替换 `author`、`journal`、`year` 等字段。)

---

## 🤝 Acknowledgements

* BUSI Dataset authors for providing the breast ultrasound dataset.
* PyTorch and related open-source libraries used in this project.

---

## 📄 License

This project is released under the **MIT License** (or your actual license).
Please see the `LICENSE` file for details.
