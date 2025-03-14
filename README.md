# Pose Quantization and Motion Prediction using Deep Learning

This repository implements a **Deep Learning-based Framework for Music-Synchronized Dance Choreography**, featuring **Pose Quantization and Motion Prediction**. The system utilizes **Vision Transformers (ViT)**, **Graph Convolutional Networks (GCN)**, **K-means Clustering**, and **Differential Evolution Optimization** to generate synchronized dance choreography based on music.
  
## **📌 Features**
✅ **SE-ViT (Vision Transformer with Squeeze & Excitation)**  
✅ **GCN-MHW (Graph Convolutional Network with Mexican Hat Wavelet)**  
✅ **Pose Quantization (K-means & VQ-VAE)**  
✅ **Music Synchronization with Beat Alignment**  
✅ **Differential Evolution Optimization for Loss Minimization**  
✅ **Trained on AIST++ Dataset**  

---

## **📂 Project Structure**
```
Pose-Quantization-and-Motion-Prediction/
│── data_preprocessing.py         # Data loading and preprocessing (AIST++)
│── se_vit.py                     # SE-ViT for pose feature extraction
│── gcn_mhw.py                    # GCN-MHW for motion prediction
│── pose_quantization.py          # K-Means and VQ-VAE-based pose quantization
│── music_synchronization.py      # Music synchronization loss function
│── differential_evolution.py     # Optimization using Differential Evolution
│── train.py                      # Model training script
│── test.py                       # Model inference and testing
│── requirements.txt              # Dependencies
│── README.md                     # Documentation
```

---

## **🚀 Installation**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/imashoodnasir/Pose-Quantization-and-Motion-Prediction.git
cd Pose-Quantization-and-Motion-Prediction
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## **📊 Dataset**
- The model is trained on the **AIST++ dataset**, a large-scale 3D human dance dataset.
- **Download AIST++**: [AIST++ Dataset](https://google.github.io/aistplusplus_dataset/)

---

## **🛠 Usage**
### **1️⃣ Preprocess Dataset**
```bash
python data_preprocessing.py
```

### **2️⃣ Train Model**
```bash
python train.py
```

### **3️⃣ Test Model**
```bash
python test.py
```

---

## **📝 Model Components**
### **1️⃣ SE-ViT (Vision Transformer)**
- Extracts **pose features** from dance video frames.
- Uses **Squeeze and Excitation (SE) blocks** for feature recalibration.

### **2️⃣ GCN-MHW (Graph Convolutional Network)**
- Converts **pose sequences into graphs**.
- Uses **Mexican Hat Wavelet** for better motion prediction.

### **3️⃣ Pose Quantization**
- **K-means clustering** groups poses into representative states.
- **VQ-VAE** (Vector Quantized Variational Autoencoders) discretizes poses.

### **4️⃣ Music Synchronization**
- Aligns pose sequences to music beats using a custom **loss function**.
- Optimized using **Differential Evolution (DE)**.

---

## **📊 Results**
### **Performance Metrics**
| Model | FID (Kinetic) ↓ | FID (Geometric) ↓ | Motion Diversity ↑ | Beat Align ↑ |
|------|-------------|-------------|----------------|-----------|
| **Proposed (log-sigmoid DE)** | **32.78** | **11.37** | **6.89** | **0.257** |
| Baseline (Li Ruilong, 2021) | 35.35 | 12.40 | 5.94 | 0.241 |
| Baseline (Li Jiaman, 2020) | 86.43 | 20.58 | 6.85 | 0.232 |

✅ **Better motion quality & diversity than existing methods!**

---

## **🔬 Research Contributions**
- **Improved Vision Transformers** for pose estimation.
- **Graph-based motion modeling** for dynamic choreography.
- **Music synchronization loss function** with **Differential Evolution**.

---

## **💡 Future Work**
- Extend to **real-time dance generation**.
- Train on **multi-dancer choreography** datasets.
- Integrate **Generative AI (GANs)** for diverse dance movements.

---

## **📜 License**
This project is licensed under the **MIT License**.

---

### 🌟 **If you found this useful, please ⭐ star the repository!**
