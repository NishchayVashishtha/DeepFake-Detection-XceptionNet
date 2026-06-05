# DeepFake Detection using XceptionNet

<div align="center">

![Parallel Coordinates Plot](/images/parcoords.gif)

**A robust deepfake detection system powered by ResNet18 that generalizes to real-world videos**

[![GitHub](https://img.shields.io/badge/GitHub-NishchayVashishtha-blue.svg)](https://github.com/NishchayVashishtha/DeepFake-Detection-XceptionNet)
[![License](https://img.shields.io/badge/License-Square%20Inc.-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.6%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red.svg)](https://pytorch.org/)

[Technical Article](https://www.dessa.com/post/deepfake-detection-that-actually-works) • [Interactive Results](http://deepfake-detection.dessa.com/projects) • [GitHub Repository](https://github.com/NishchayVashishtha/DeepFake-Detection-XceptionNet)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements a **state-of-the-art deepfake detection system** using PyTorch and ResNet18 architecture. Our research demonstrates that existing models trained solely on FaceForensics++ dataset fail to generalize to real-world deepfakes found on platforms like YouTube.

### Key Contributions

✅ **Generalization Analysis**: Demonstrated that FaceForensics++ trained models don't generalize to real-world manipulations  
✅ **Augmented Dataset**: Collected and integrated real YouTube deepfake data for improved robustness  
✅ **Production-Ready**: Provides a practical solution for detecting various manipulation techniques  
✅ **Comprehensive Experiments**: Large-scale hyperparameter tuning using Atlas experiment manager

---

## ⭐ Key Features

- **Multi-Dataset Training**: Supports FaceForensics++, YouTube augmented data, and combined datasets
- **Transfer Learning**: Fine-tuned ResNet18 pre-trained on ImageNet
- **Advanced Data Augmentation**: Random flips, affine transformations, color jittering, and perspective changes
- **Face Detection & Cropping**: Automatic face detection using dlib for preprocessing
- **Mixed Precision Training**: Leverages NVIDIA Apex for faster training
- **Experiment Tracking**: Integrated with Atlas for managing experiments and visualizations
- **Hyperparameter Optimization**: Automated random search across 140+ configurations
- **Real-time Visualization**: Training metrics, predictions, and model performance plots

---

## 🏗️ Architecture

### Model Pipeline

```
Input Video → Frame Extraction → Face Detection → Face Cropping → ResNet18 → Classification (Real/Fake)
```

### Neural Network Architecture

- **Base Model**: ResNet18 (pre-trained on ImageNet)
- **Custom Head**: Optional hidden layer with dropout, batch normalization
- **Output**: Binary classification (Real vs Fake)
- **Optimizer**: Adam with OneCycleLR scheduling
- **Loss Function**: Cross Entropy Loss
- **Mixed Precision**: NVIDIA Apex O1 optimization

### Data Augmentation Strategy

**Training Augmentations:**
- Random horizontal flips
- Random affine transformations (rotation, scale, shear)
- Random perspective transformations
- Color jitter (brightness, contrast, saturation)
- Random erasing for occlusion robustness
- ImageNet normalization

**Validation:** Standard normalization only

---

## 📁 Project Structure

```
DeepFake-Detection/
├── main.py                      # Main training script
├── train.py                     # Training loop implementation
├── model.py                     # ResNet18 model definition
├── data_loader.py               # Custom dataset and dataloader
├── utils.py                     # Utility functions (visualization, metrics)
├── hparams_search.py            # Hyperparameter search script
├── faceforensics_download.py    # FaceForensics++ download utility
├── download_T.sh                # Dataset download script
├── restructure_data.sh          # Data restructuring script
├── job.config.yaml              # Atlas job configuration
├── custom_docker_image/
│   ├── Dockerfile               # Docker environment setup
│   └── requirements.txt         # Python dependencies
└── images/                      # Result visualizations
    ├── parcoords.gif
    ├── model1.png - model33.png
    └── ...
```

### Core Modules

| File | Description |
|------|-------------|
| `main.py` | Entry point for training with parameter loading and initialization |
| `train.py` | Training loop, validation, metrics logging, and checkpointing |
| `model.py` | ResNet18 architecture with custom classification head |
| `data_loader.py` | Custom PyTorch Dataset for loading and preprocessing frames |
| `utils.py` | Face detection, cropping, visualization, and metrics computation |
| `hparams_search.py` | Random hyperparameter search across 140 configurations |

---

## 💻 Requirements

### System Requirements

- **RAM**: ≥ 32GB
- **GPU**: ≥ 1 NVIDIA GPU (CUDA compatible)
- **Storage**: ~500GB for datasets
- **OS**: Linux (Ubuntu 18.04+ recommended)

### Software Dependencies

```
Python >= 3.6
PyTorch >= 1.x
CUDA >= 10.0
nvidia-docker
ffmpeg
```

### Python Packages

```
torch
torchvision
numpy==1.17.3
dlib==19.15.0
opencv-python
Pillow==6.2.0
scikit-learn==0.21.3
matplotlib==3.1.1
tqdm==4.25.0
apex (NVIDIA mixed precision training)
foundations (Atlas experiment manager)
```

---

## 🚀 Installation

### Step 1: Install System Dependencies

```bash
# Install NVIDIA Docker
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Install FFmpeg
sudo apt install ffmpeg
```

### Step 2: Clone Repository

```bash
git clone https://github.com/NishchayVashishtha/DeepFake-Detection-XceptionNet.git
cd DeepFake-Detection-XceptionNet
```

### Step 3: Install Atlas

```bash
pip install foundations-atlas
foundations init
```

### Step 4: Build Docker Image

```bash
cd custom_docker_image
nvidia-docker build . -t atlas_ff
cd ..
```

**Note**: If you change the docker image name, update `job.config.yaml` line 16 accordingly.

---

## 📊 Dataset Setup

### Dataset Sources

1. **FaceForensics++**: Research dataset with controlled manipulations
   - [Request Access](https://github.com/ondyari/FaceForensics/#access) via Google Form
   - Contains: DeepFakes, Face2Face, FaceSwap, NeuralTextures

2. **YouTube Augmented**: Real-world deepfakes collected from YouTube
   - [Download from S3](https://deepfake-detection.s3.amazonaws.com/augment_deepfake.tar.gz)

### Automated Setup

```bash
# After receiving FaceForensics++ download script
bash restructure_data.sh faceforensics_download.py
```

### Expected Directory Structure

```
datasets/
├── T_deepfake/                 # FaceForensics++ raw data
│   ├── manipulated_sequences/
│   │   ├── DeepFakeDetection/
│   │   ├── Deepfakes/
│   │   ├── Face2Face/
│   │   ├── FaceSwap/
│   │   └── NeuralTextures/
│   └── original_sequences/
│       ├── actors/
│       └── youtube/
│
├── base_deepfake/             # FaceForensics++ processed
│   ├── fake/frames/
│   ├── real/frames/
│   └── val/
│       ├── fake/
│       └── real/
│
├── augment_deepfake/          # YouTube dataset
│   ├── fake/frames/
│   ├── real/frames/
│   └── val/
│       ├── fake/
│       └── real/
│
├── both_deepfake/             # Combined dataset
│   ├── fake/frames/
│   ├── real/frames/
│   └── val/
│       ├── fake/
│       └── real/
│
└── precomputed/               # Cached preprocessed faces (auto-generated)
```

### Configuration

Update `job.config.yaml` with your dataset path:

```yaml
worker:
  volumes:
    /absolute/path/to/datasets/:  # Change this path
      bind: /datasets/
      mode: rw
```

---

## 🎮 Usage

### Single Training Run

```bash
python main.py
```

This will:
1. Load parameters from Atlas (or defaults)
2. Create train/validation dataloaders
3. Initialize ResNet18 model
4. Train for specified epochs
5. Save best model checkpoint
6. Generate visualization plots

### Training with Custom Parameters

Modify parameters in `main.py` or through Atlas:

```python
params = {
    'batch_size': 512,
    'n_epochs': 20,
    'max_lr': 0.00005,
    'weight_decay': 0.1,
    'dropout': 0.8,
    'use_lr_scheduler': 1,
    'use_hidden_layer': 1,
    'train_data': 'both',  # 'base', 'augment', or 'both'
    'sample_ratio': 1.0,
    'pct_start': 0.3
}
```

### Model Training Options

**Dataset Selection:**
- `base`: FaceForensics++ only
- `augment`: YouTube dataset only
- `both`: Combined dataset (recommended)

**Architecture Options:**
- `use_hidden_layer=0`: Direct classification (512 → 2)
- `use_hidden_layer=1`: Hidden layer with BatchNorm (512 → 256 → 2)

---

## 🔧 Hyperparameter Tuning

Run automated hyperparameter search across 140 configurations:

```bash
python hparams_search.py
```

### Search Space

| Parameter | Range/Options |
|-----------|---------------|
| `batch_size` | [256, 512, 1024] |
| `n_epochs` | [15, 20, 25] |
| `max_lr` | [0.00003, 0.00007] |
| `weight_decay` | [0.01, 0.3] |
| `dropout` | [0.75, 0.8, 0.9] |
| `pct_start` | [0.3, 0.5] |
| `use_lr_scheduler` | [0, 1] |
| `use_hidden_layer` | [0, 1] |
| `sample_ratio` | [0.1, 0.25, 0.5, 0.75, 1.0] |

### Monitoring Experiments

Access Atlas web interface to:
- View real-time training progress
- Compare hyperparameter configurations
- Download trained models
- Analyze parallel coordinates plots

---

## 📈 Results

### Model Performance Comparison

#### 1️⃣ Trained on FaceForensics++ Only

❌ **Problem**: Fails to detect real-world YouTube deepfakes  
✅ **Strength**: Good at detecting known manipulation techniques (Face2Face, FaceSwap)

![Model 1](/images/model1.png)
![Model 1 Detail](/images/model11.png)

**Interpretation**: `[Real_Prob, Fake_Prob] | Prediction | True_Label`

---

#### 2️⃣ Trained on YouTube Dataset Only

✅ **Strength**: Excellent at detecting real-world deepfakes  
❌ **Problem**: Misses advanced manipulations like NeuralTextures

![Model 2](/images/model2.png)
![Model 2 Detail](/images/model22.png)

---

#### 3️⃣ Trained on Combined Dataset (Recommended)

✅ **Best Overall**: Detects both research dataset manipulations AND real-world deepfakes  
✅ **Robust**: Generalizes across multiple manipulation techniques

![Model 3](/images/model3.png)
![Model 3 Detail](/images/model33.png)

---

### Key Findings

| Model | FaceForensics++ Acc | YouTube Acc | Generalization |
|-------|---------------------|-------------|----------------|
| Base Only | High | Low | ❌ Poor |
| Augment Only | Medium | High | ⚠️ Limited |
| **Combined** | **High** | **High** | ✅ **Excellent** |

### Metrics Tracked

- **Loss**: Cross-entropy loss on train/validation
- **Accuracy**: Classification accuracy (Real vs Fake)
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: True/False positives and negatives

---

## 🛠️ Advanced Features

### Mixed Precision Training

Uses NVIDIA Apex for 2x faster training with lower memory:

```python
model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
```

### Learning Rate Scheduling

OneCycleLR for faster convergence:

```python
scheduler = lr_scheduler.OneCycleLR(
    optimizer, max_lr=params['max_lr'],
    epochs=params['n_epochs'],
    steps_per_epoch=len(train_dl),
    pct_start=params['pct_start']
)
```

### Face Preprocessing Pipeline

```python
Image → Grayscale → Face Detection (dlib) → Bounding Box → Crop → Resize → Cache
```

Cached preprocessed faces save computation time in subsequent epochs.

---

## 📝 Citation

If you use this work in your research, please cite:

```
Dessa (Square, Inc.). "DeepFake Detection That Actually Works."
https://www.dessa.com/post/deepfake-detection-that-actually-works
```

---

## 🤝 Contributing

We welcome contributions to improve deepfake detection technology!

### How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Share Your Data

Help improve detection accuracy by sharing:
- Real deepfake videos from the wild
- Novel manipulation techniques
- Edge cases where the model fails

📧 Contact: foundations@dessa.com

---

## 📞 Support

For questions or issues:
- Open a [GitHub Issue](https://github.com/NishchayVashishtha/DeepFake-Detection-XceptionNet/issues)
- Read the [Technical Article](https://www.dessa.com/post/deepfake-detection-that-actually-works)
- Explore [Interactive Results](http://deepfake-detection.dessa.com/projects)
- Contact: [Nishchay Vashishtha](https://github.com/NishchayVashishtha)

---

## 📜 License

© 2020 Square, Inc.

ATLAS, DESSA, the Dessa Logo, and others are trademarks of Square, Inc. All third party names and trademarks are properties of their respective owners and are used for identification purposes only.

See [LICENSE](LICENSE) for more details.

---

## 🙏 Acknowledgments

- FaceForensics++ team for the benchmark dataset
- Dessa (Square Inc.) for the original research and implementation
- Atlas team for experiment management infrastructure
- PyTorch and dlib communities
- All contributors and researchers in deepfake detection

## 👨‍💻 Author

**Nishchay Vashishtha**
- GitHub: [@NishchayVashishtha](https://github.com/NishchayVashishtha)
- Repository: [DeepFake-Detection-XceptionNet](https://github.com/NishchayVashishtha/DeepFake-Detection-XceptionNet)

---

<div align="center">

**⚡ Built with PyTorch • Powered by Atlas • Secured by AI ⚡**

[⬆ Back to Top](#deepfake-detection-using-deep-learning)

</div>
