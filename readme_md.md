# LightBASNeSt: A Lightweight Boundary-Aware Split-Attention Network for Black Gram Leaf Disease Classification

[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation of the paper: **"LightBASNeSt: A Lightweight Boundary-Aware Split-Attention Network for Black Gram Leaf Disease Classification"**. 

LightBASNeSt is a highly efficient, hierarchical deep learning model designed for the precise multi-class classification of leaf diseases in Black Gram (*Vigna mungo*). It balances high predictive accuracy with a compact computational footprint, making it highly suitable for mobile and low-power edge deployment in precision agriculture.

## 🌟 Key Features & Contributions

*   **RadixASEPConv:** A novel split-attention feature extraction module that utilizes depthwise and pointwise convolutions alongside adaptive dual-pooling (GAP + GMP) to capture complementary feature representations at a minimal computational cost.
*   **Local Boundary Spatial Attention (LBSA):** A specialized attention module that emphasizes lesion boundaries and high-frequency spatial characteristics using learned spatial attention paired with a fixed Laplacian edge operator.
*   **GLocalHead:** A unique classification head that fuses global contextual information with localized ($2\times2$) disease representations to prevent the loss of fine-grained disease patterns.
*   **Lightweight & Efficient:** Achieves state-of-the-art accuracy with only **3.90 million parameters** and **0.470 GFLOPs**.
*   **Explainable AI (XAI):** Built-in support for Grad-CAM and Score-CAM to visually validate biological plausibility.

---

## 📊 Model Performance

LightBASNeSt was extensively evaluated on two publicly available Black Gram leaf datasets as well as a merged configuration.

| Dataset | Classes | Images | Accuracy | F1-Score | PR-AUC | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dataset-1 (D1)** | 5 | 4,038 | **99.51%** | 0.9940 | 0.9997 | 0.9988 |
| **Dataset-2 (D2)** | 5 | 1,007 | **98.02%** | 0.9802 | 0.9822 | 0.9949 |
| **Merged (D1 + D2)**| 7 | 5,045 | **98.61%** | 0.9862 | 0.9957 | 0.9976 |

**Computational Profile:**
*   **Parameters:** 3.90 M
*   **GFLOPS:** 0.470
*   **Inference Time:** 4.06 s (Batch Size 32, NVIDIA RTX 4060)

---

## 📂 Dataset Preparation

This study utilizes two public datasets available via Mendeley Data:
1.  **Dataset-1 (BPLD):** Contains classes for Cercospora leaf spot, Healthy, Insect, Leaf Crinkle, and Yellow Mosaic.
2.  **Dataset-2 (IDBGL):** Contains field-captured images for Anthracnose, Healthy, Leaf Crinkle, Powdery Mildew, and Yellow Mosaic.

Organize your downloaded datasets in the following structure before training:

```text
Dataset/
├── Original/
│   ├── Anthracnose/
│   ├── Cercospora_Spot/
│   ├── Healthy/
│   ├── Insect/
│   ├── Leaf_Crinkle/
│   ├── Powdery_Mildew/
│   └── Yellow_Mosaic/
```

---

## ⚙️ Installation & Requirements

**Hardware Requirements:** GPU acceleration (NVIDIA RTX 4060 or better recommended).
**Software Requirements:** Python 3.12, CUDA 12.6.

1. Clone the repository:
   ```bash
   git clone https://github.com/SamiLzabid/BlackGrm_Paper.git
   cd BlackGrm_Paper
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure you install the correct PyTorch version compatible with your CUDA toolkit).*

---

## 🚀 Usage

### 1. Training the Model
To train LightBASNeSt from scratch on Dataset-1:
```bash
python train.py --data_dir ./Dataset/Original --dataset D1 --epochs 30 --batch_size 32 --lr 5e-4
```

### 2. Evaluating the Model
To evaluate a trained checkpoint on the test set:
```bash
python evaluate.py --data_dir ./Dataset/Original --weights ./checkpoints/best_model.pth
```

### 3. Generating XAI Visualizations (Grad-CAM / Score-CAM)
To generate saliency maps for model interpretability:
```bash
python explain.py --image_path ./sample_images/yellow_mosaic.jpg --weights ./checkpoints/best_model.pth --method scorecam
```

---

## 🏗️ Repository Structure

```text
BlackGrm_Paper/
│
├── models/
│   ├── lightbasnest.py       # Core architecture, RadixASEPConv, LBSA, GLocalHead
│   └── building_blocks.py    # Utility layers (LayerScale, DropPath)
│
├── utils/
│   ├── dataset.py            # Data loading, augmentation, and preprocessing
│   ├── metrics.py            # MCC, PR-AUC, F1-score calculations
│   └── visualization.py      # Grad-CAM, Score-CAM, Confusion Matrix plots
│
├── train.py                  # Main training script (AdamW, Cosine Annealing)
├── evaluate.py               # Evaluation script
├── explain.py                # Script for generating visual explanations
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 📝 Citation

If you find this code or research helpful in your work, please cite our paper:

```bibtex
@article{zabid2026lightbasnest,
  title={LightBASNeSt: A Lightweight Boundary-Aware Split-Attention Network for Black Gram Leaf Disease Classification},
  author={Zabid, Sami Al and Bhuiyan, Md. Shakil and Karim, Md Anwarul and Mahmud, S.K Jihad and Polash, Md. Saymon Hosen and Uddin, Jia and Alturki, Abdulrahman S.},
  journal={TBD},
  year={2026},
  publisher={TBD}
}
```

## 👥 Authors & Acknowledgements
**Authors:** Sami Al Zabid, Md. Shakil Bhuiyan, Md Anwarul Karim, S.K Jihad Mahmud, Md. Saymon Hosen Polash, Jia Uddin, and Abdulrahman S. Alturki.

**Affiliations:** 
* East West University, Dhaka, Bangladesh 
* Woosong University, Daejeon, Republic of Korea
* Qassim University, Buraydah, Saudi Arabia

*This research was supported by the Deanship of Graduate Studies and Scientific Research at Qassim University.*