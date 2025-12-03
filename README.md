# ThermoVisSR: Multi-Scale Transformer Network for Super-Resolution of Visible and Thermal Air Images

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-Intelligent%20Systems%20with%20Applications-blue)](https://doi.org/10.1016/j.iswa.2024.200429)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-green.svg)](http://creativecommons.org/licenses/by-nc-nd/4.0/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.1.0-brightgreen.svg)](https://developer.nvidia.com/cuda-toolkit)

*Official PyTorch implementation of "Multi-scale transformer network for super-resolution of visible and thermal air images"*

**[Paper](https://doi.org/10.1016/j.iswa.2024.200429)** | **[Dataset Request]** | **[Demo]**

</div>

---

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Visual Results](#visual-results)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Quantitative Results](#quantitative-results)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## 🔍 Overview

**ThermoVisSR** is a state-of-the-art multi-scale texture transformer designed for The super-resolution of both **visible (RGB)** and **thermal (infrared)** images of Mini/Micro UAVs. This work addresses the unique challenges of detecting and recognizing small aerial objects in video surveillance applications by leveraging dual-stream processing and reference-based super-resolution techniques.

### 🎯 Problem Statement

Traditional super-resolution methods struggle with:
- ❌ Small objects like Mini/Micro UAVs with limited texture information
- ❌ Separate processing of visible and thermal streams
- ❌ Resolution disparities between input and reference images
- ❌ Preservation of fine details while maintaining color accuracy

### ✅ Our Solution

ThermoVisSR introduces a novel approach that:
- ✓ **Fuses visible and thermal information** to leverage the advantages of both modalities
- ✓ **Separates high-frequency (HF) and low-frequency (LF) processing** for better reconstruction
- ✓ **Applies multi-scale texture transformers** for accurate correspondence matching
- ✓ **Achieves superior performance** on both quantitative metrics and qualitative assessments

---

## 🏗️ Architecture

### Overall Network Architecture

<div align="center">
  <img src="figures/model.png" alt="ThermoVisSR Architecture" width="100%"/>
  <p><i>Figure 1: ThermoVisSR architecture showing the dual-stream processing of visible and thermal images with multi-scale texture transformers.</i></p>
</div>


---

## ✨ Key Features

| Feature | Description | Benefit |
|---------|-------------|---------|
| 🔄 **Dual-Stream Processing** | Simultaneous enhancement of RGB and IR images | Leverages complementary information |
| 🔍 **Multi-Scale HFTT** | Operates across multiple resolutions | Better correspondence matching |
| 🎛️ **Separable Soft Decoder** | Channel-specific detail enhancement | Optimal for heterogeneous data |
| 🎨 **Fusion Backbone** | Preserves color and body form | Artifact-free reconstruction |
| 🌓 **Lighting Robustness** | Excellent performance in low-light | Reliable 24/7 surveillance |

---

## 🎨 Visual Results

### Visible Image Super-Resolution (4× Scale)

<div align="center">
  <img src="figures/SR4.png?v=2" alt="Visible SR Results - DJI Phantom" width="100%"/>
  <p><i>Figure 5: Visual comparison of SR approaches on visible image of DJI-Phantom4Pro (4× resolution)</i></p>
</div>

### Thermal Image Super-Resolution (4× Scale)

<div align="center">
  <img src="figures/SR4-THR.png" alt="Thermal SR Results - DJI Phantom" width="100%"/>
  <p><i>Figure 6: Visual comparison on thermal image of DJI-Phantom4Pro (4× resolution)</i></p>
</div>

### 8× Super-Resolution Results

<div align="center">
  <img src="figures/SR8.png" alt="Visible 8× SR" width="100%"/>
  <p><i>Figure 7: Visual comparison of SR methods on visible images at 8× resolution</i></p>
</div>

<div align="center">
  <img src="figures/SR8-THR.png" alt="Thermal 8× SR" width="100%"/>
  <p><i>Figure 8: Visual comparison of SR methods on thermal images at 8× resolution</i></p>
</div>

---

## 🚀 Installation

### Prerequisites
```bash
Python 3.7+
CUDA 11.1.0
cuDNN v8.2.2
```

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ThermoVisSR.git
cd ThermoVisSR

# Create conda environment
conda create -n thermovssr python=3.8
conda activate thermovssr

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Dataset

### Mini/Micro UAVs Co-registered Dataset

**First dataset of its kind** containing co-registered visible and thermal images of Mini/Micro UAVs.

> ⚠️ **Dataset Availability**: The dataset is available upon request for academic research purposes only. Please contact the authors to request access.

#### Dataset Characteristics

| Property | Details |
|----------|---------|
| **UAV Types** | DJI Matrice 600 Pro, DJI Phantom 4 Pro, DJI Mavic |
| **Environments** | Urban and rural backgrounds |
| **Capture Conditions** | Various altitudes, speeds, lighting |
| **Training Set** | 736 pairs (3,680 after augmentation) |
| **Test Set** | 146 pairs |
| **Image Size** | 360×360 pixels (HR) |
| **Channels** | 3 visible (RGB) + 1 thermal (IR) |

#### EO/IR Imaging System

- **System**: DH-TPC-PT8621C (Dahua Technology)
- **Features**: Co-registered visible and thermal cameras
- **Recording**: Simultaneous dual-band capture

#### Dataset Structure

```
DRONES/
├── train/
│   ├── input_vis/
│   ├── input_thr/     
│   ├── ref_vis/       
│   ├── ref_thr/      
└── test/
    ├── input_vis/    
    ├── input_thr/     
```

#### 📋 How to Request Dataset Access

To request access to the **Mini/Micro UAVs Co-registered Dataset**, please:

1. **Send an email** to: [HEDI.FEKI.doc@enetcom.usf.tn](mailto:HEDI.FEKI.doc@enetcom.usf.tn)
2. **Include the following information**:
   - Your name and affiliation (university/research institution)
   - Brief description of your research project
   - Intended use of the dataset
   - Confirmation that the dataset will be used for academic/research purposes only

3. **Subject line**: `Request for Mini/Micro UAVs Co-registered Dataset Access`

**Note**: The dataset is shared under specific terms and conditions for research purposes only. Commercial use is not permitted.


---

## 🎓 Training

### Start Training

```bash
sh train.sh
```

### Training Time & Resources

| Configuration | Time | GPU Memory |
|--------------|------|------------|
| Single Tesla T4 | ~15 hours | 16 GB |
| Single A100 | ~8 hours | 40 GB |

---

## 📈 Evaluation

### Run Evaluation

```bash
sh eval.sh
```

### Evaluation Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **PSNR** | Peak Signal-to-Noise Ratio | Higher is better |
| **SSIM** | Structural Similarity Index | [0, 1], higher is better |
| **MS-SSIM** | Multi-Scale SSIM | [0, 1], higher is better |
| **VIF** | Visual Information Fidelity | Higher is better |

---

## 🏆 Quantitative Results

### Performance Comparison (4× Scale Factor)

| Method | Type | PSNR ↑ | SSIM ↑ | MS-SSIM ↑ | VIF ↑ |
|--------|------|--------|--------|-----------|-------|
| SRCNN | SISR | 28.43 | 0.870 | 0.890 | 0.600 |
| SRResNet | SISR | 28.91 | 0.897 | 0.910 | 0.630 |
| SRGAN | SISR | 29.43 | 0.898 | 0.913 | 0.640 |
| ESRGAN | SISR | 29.71 | 0.923 | 0.935 | 0.680 |
| RankSRGAN | SISR | 30.21 | 0.930 | 0.940 | 0.710 |
| DAT | SISR | 31.02 | 0.945 | 0.950 | 0.720 |
| SRNTT-rec | RefSR | 31.38 | 0.949 | 0.955 | 0.755 |
| TTSR-rec | RefSR | 31.88 | 0.950 | 0.958 | 0.760 |
| C2-Matching | RefSR | 32.37 | 0.957 | 0.963 | 0.790 |
| DATSR | RefSR | 32.96 | 0.961 | 0.966 | 0.800 |
| **ThermoVisSR** | **RefSR** | **34.88** | **0.970** | **0.975** | **0.825** |

**Improvements over SOTA:**
- 📈 **+1.92 dB PSNR** vs. DATSR
- 📈 **+0.009 SSIM** vs. DATSR
- 📈 **+0.009 MS-SSIM** vs. DATSR
- 📈 **+0.025 VIF** vs. DATSR

### Performance Comparison (8× Scale Factor)

| Method | PSNR ↑ | SSIM ↑ | MS-SSIM ↑ | VIF ↑ |
|--------|--------|--------|-----------|-------|
| DAT | 28.25 | 0.903 | 0.915 | 0.630 |
| TTSR-rec | 29.20 | 0.931 | 0.935 | 0.660 |
| C2-Matching | 29.37 | 0.935 | 0.945 | 0.680 |
| DATSR | 29.86 | 0.941 | 0.950 | 0.690 |
| **ThermoVisSR** | **31.24** | **0.950** | **0.960** | **0.700** |


### Performance Under Different Lighting Conditions

| Condition | PSNR ↑ | SSIM ↑ | MS-SSIM ↑ | VIF ↑ |
|-----------|--------|--------|-----------|-------|
| **High-Light** | **34.30** | **0.970** | **0.980** | **0.850** |
| **Low-Light** | **32.85** | **0.967** | **0.972** | **0.790** |

---

## 💡 Key Advantages

### ✅ Technical Strengths

1. **🎯 Superior Texture Preservation**
   - Selective HF texture transfer maintains fine details
   - No color distortion or artifacts

2. **🎨 Color & Body Form Accuracy**
   - Fusion Backbone preserves LF information
   - Realistic reconstruction of UAV components

3. **🔄 Multi-Modal Fusion**
   - Leverages complementary visible and thermal information
   - Enhanced detection capabilities day and night

4. **📐 Multi-Scale Processing**
   - HFTT operates at multiple resolutions
   - Better correspondence matching for small objects

5. **🌙 Lighting Robustness**
   - Consistent performance in low-light conditions
   - Minimal degradation compared to high-light

### 🎯 Application Benefits

- ✈️ **UAV Detection**: Enhanced identification of Mini/Micro UAVs
- 🔍 **Sky Surveillance**: 24/7 monitoring capabilities
- 🎥 **Video Enhancement**: Real-time processing potential
- 🛡️ **Security Systems**: Improved threat detection

---

## 📝 Citation

If you find this work useful for your research, please cite:

```bibtex
@article{fkih2024thermovssr,
  title={Multi-scale transformer network for super-resolution of visible and thermal air images},
  author={Fkih, H{\`e}di and Kallel, Abdelaziz and Chtourou, Zied},
  journal={Intelligent Systems with Applications},
  volume={23},
  pages={200429},
  year={2024},
  publisher={Elsevier},
  doi={10.1016/j.iswa.2024.200429}
}
```

---

## 🙏 Acknowledgments

This work is funded by:
- **🇹🇳 Tunisian Ministry of National Defense**
- **🏛️ Digital Research Center of Sfax**
- **SM@RTS Laboratory** (Signals systeMs aRtificial intelligence and neTworkS Laboratory)

---

## 👥 Authors

<table>
  <tr>
    <td align="center">
      <strong>Hèdi Fkih</strong><br>
      <i>ip-label, Tunis</i><br>
      📧 <a href="mailto:HEDI.FEKI.doc@enetcom.usf.tn">Email</a>
    </td>
    <td align="center">
      <strong>Abdelaziz Kallel</strong><br>
      <i>Digital Research Center of Sfax</i><br>
      📧 <a href="mailto:abdelaziz.kallel@crns.rnrt.tn">Email</a>
    </td>
    <td align="center">
      <strong>Zied Chtourou</strong><br>
      <i>School of Aeronautical Specialties</i><br>
      📧 <a href="mailto:ziedchtourou@gmail.com">Email</a>
    </td>
  </tr>
</table>

---

## 📄 License

This project is licensed under the **CC BY-NC-ND 4.0** License.

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc-nd/4.0/)

---

## 📧 Contact

For questions, collaborations, or dataset access:
- 📝 Open an issue on GitHub
- 📬 Contact: [HEDI.FEKI.doc@enetcom.usf.tn](mailto:HEDI.FEKI.doc@enetcom.usf.tn)

---

## 📚 Related Resources

- **Paper**: [Intelligent Systems with Applications](https://doi.org/10.1016/j.iswa.2024.200429)
- **Institution**: [ip-label africa](https://ip-label.com/)
- **Laboratory**: [SM@RTS Lab](https://smarts.tn/)

# ThermoVisSR
ThermoVisSR: Multi-Scale Transformer Network for Super-Resolution of Visible and Thermal Air Images

