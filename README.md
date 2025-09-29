# ContourFD-Net
ContourFD-Net: Finite Difference Gradient-Guided Contour-Aware Attention Network for Medical Image Segmentation

## How To Use

#### Dependencies

- OS Requirements:
    - Linux (Ubuntu/Debian) (Windows is not officially supported)
    - CUDA >=11.6
    - cuDNN 8.2.4
    - Python >= 3.8
    - PyTorch >=1.12.1
- Our environment:
    - OS: Debian 12 (bookworm)
    - GPU: NVIDIA 3090 / NVIDIA 3080ti
    - CUDA 11.8
    - cuDNN 8.9.7
    - Python 3.8.16
    - PyTorch 2.0.1
    
- Clone this repository 
```bash
git clone https://github.com/tpnam0901/QTSeg.git
cd qtseg
```
- Create a conda environment and install requirements
```bash
conda create -n qtseg python=3.8.16 -y
conda activate qtseg 
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```
or with conda environment
```bash
conda env create -f environment.yml
```

#### Preprocessing dataset

##### Breast Ultrasound Images Dataset (BUSI)
- Dataset used in this project is [BUSI](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) - [link_backup](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset).

- After downloading the dataset, you need to extract the dataset and put it in the data folder. The folder structure should be as follows:
```
# Note: do not change the filename of the images
- working/dataset/BUSI # or any path you want
    - benign
        - *.png
    - malignant
        - *.png
    - normal
        - *.png
```
- First, before splitting the dataset into 5 folds, we need to merge any sample that has more than 1 label file. This can be done by running the following command:
```bash
cd src/tools && python busi_01_combine_masks.py --data_root ../working/dataset/BUSI/
```
- Then, we can split the dataset into 5 folds by running the following command:
```bash
cd src/tools && python busi_02_split_folds.py --data_root ../working/dataset/BUSI/
```
- The dataset is sorted by the names of the images before splitting. The samples are chosen by slicing the array with a size of the number of samples in each fold. So, the same fold index will have the same samples in different runs. After the process, the dataset will be split into 5 folds and saved in the `working/dataset/BUSI/folds` folder as used in the paper.


#### Configuration & Training

- Before starting training, you need to download pre-trained models from [fpn-nano.pth](https://github.com/tpnam0901/QTSeg/releases/download/v0.2.0/fpn-nano.pth) and put them in the `src/networks/pretrained` folder. In case the link is broken, you can download the pre-trained models from the assets in the [release page](https://github.com/tpnam0901/QTSeg/releases). The structure of your folder should be as follows:
```bash
src
├── networks
    ├── pretrained
        ├── fpn-nano.pth
```

- The configuration file is located in the `configs` folder. You can modify the configuration file to suit your needs. For example, you can change the dataset path, the number of classes, the batch size, the number of epochs, etc.

- Train ISIC dataset
```bash
cd src && python train.py -cfg configs/ISIC.py
```

- Train BUSI dataset
```bash
cd src && python train.py -cfg configs/BUSI.py
```

- Train BKAI dataset
```bash
cd src && python train.py -cfg configs/BKAI.py
```

- Train DSB2018 dataset
```bash
cd src && python train.py -cfg configs/DSB2018.py
```

- Train FIVES dataset
```bash
cd src && python train.py -cfg configs/FIVES.py
```

- You can visualize the training process by running the following command:
```bash
# This is the checkpoint folder that contains `mlruns` folder
cd src/working/checkpoint/QTSeg/<DATASET_NAME>/ && mlflow server -p 5000
```

#### Evaluation & Inference

- After training, you will have the checkpoints saved in the `working/checkpoints` folder which contains the model weights in the `.pt` format and the `.json` file containing the configuration of the model. You can evaluate the model by running the following command:
```bash
# For all checkpoints
cd src && python eval.py -cfg working/checkpoint/QTSeg/20241120-223413/cfg.json
or
# For specific checkpoint
cd src && python eval.py -cfg working/checkpoint/QTSeg/20241120-223413/cfg.json --ckpt working/checkpoint/QTSeg/20241120-223413/weight_best_iou.pt 
```

- For inference, you can run the following command:
```bash
# For all checkpoints
cd src && python infer.py -cfg working/checkpoint/QTSeg/20241120-223413/cfg.json --input_dir path/to/input_dir --output_dir path/to/output_dir
# For specific checkpoint
cd src && python infer.py -cfg working/checkpoint/QTSeg/20241120-223413/cfg.json --ckpt working/checkpoint/QTSeg/20241120-223413/weight_best_iou.pt --input_dir path/to/input_dir --output_dir path/to/output_dir
```

- We also provide pre-trained models for ISIC and FIVES datasets. you can download the pre-trained models from the assets in the [release page](https://github.com/tpnam0901/QTSeg/releases/tag/assets) and put them in the `src/working/checkpoints` folder. The structure of your folder should be as follows:
```bash
src
├── working
    ├── checkpoints
        ├── ISIC
            ├── mlruns
            ├── QTSeg
                ├── FPNEncoderMaskDecoder
                    20250122-160324
                        ├── cfg.json
                        ├── *.pt
        ├── FIVES
        ...
```

- Some results:
<p align="center">
    <img src="./assets/performance.png"/>
</p>
<p align="center">
<img src="./assets/isic_viz.png" width="800"/>
</p>
<p align="center">
    <img src="./assets/bkai_viz.png" width="400"/>
    <img src="./assets/dsb2018_viz.png" width="400"/>
</p>
<p align="center">
    <img src="./assets/fives_viz.png" width="800"/>
</p>
