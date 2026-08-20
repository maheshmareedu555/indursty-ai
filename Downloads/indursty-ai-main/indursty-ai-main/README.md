# 🏭 Smart Manufacturing AI Agent - Industrial Defect Detection & Quality Inspection

An autonomous, multi-model Industrial AI Agent and Web Application for real-time metal casting defect detection, automated quality control inspection, visual explainability (XAI), and dataset diagnostic auditing.

---

## 🌟 Executive Summary

In smart manufacturing lines, high-speed automated visual quality inspection is critical to minimizing production scrap and preventing defective components from reaching assembly. This project integrates three complementary computer vision architectures—**YOLOv11**, **EfficientNet-B3**, and **Vision Transformer (ViT-B/16)**—managed by an **Agentic Diagnostic Engine**. 

The system provides:
- **Real-time Defect Localization**: Bounding boxes around casting surface defects (porosity, blowholes, cracks).
- **Pass/Fail Classification**: High-precision overall component quality rating.
- **Explainable AI (XAI)**: Dual-layer visual explainability using **Grad-CAM heatmaps** and **ViT patch attention maps**.
- **Multi-Model Consensus Engine**: Automated confidence scoring, multi-model voting, and root-cause manufacturing recommendations.
- **Interactive Web Dashboard**: Modern FastAPI-backed dark glassmorphism dashboard with interactive Chart.js analytics and live inspection tools.

---

## 📊 Model Architecture & Benchmark Performance

Each model in the ensemble addresses a specific operational requirement in the inspection pipeline:

| Model | Primary Operational Role | Accuracy | Precision | Recall | F1-Score | Avg Latency | Model Parameters |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B3** | High-Accuracy Quality Classification & Grad-CAM Heatmaps | **98.50%** | **98.40%** | **98.60%** | **0.9850** | 28.5 ms | 12.2M |
| **Vision Transformer (ViT-B/16)** | Global Context Patch Self-Attention Analysis | **97.90%** | **97.80%** | **98.00%** | **0.9790** | 42.1 ms | 86.6M |
| **YOLOv11** | Real-Time Defect Detection & Bounding Box Localization | **96.80%** | **96.50%** | **97.10%** | **0.9680** | **14.2 ms** | 2.6M |

*Note: YOLOv11 achieves an additional **mAP@50 of 94.2%** for single-class defect bounding box localization.*

---

## 📁 Dataset & Audit Metrics

The underlying industrial casting dataset underwent automated cleaning, perceptual hash deduplication, and file verification:

- **Total Audited Raw Images**: 8,648
- **Clean Unique Samples**: 8,584
- **Deduplication Audit**: 64 duplicate images identified & removed (0 corrupt files)
- **Dataset Partitioning (70% Train / 15% Val / 15% Test)**:
  - **Train Set**: 3,494 Defective | 2,514 OK (6,008 total)
  - **Val Set**: 748 Defective | 538 OK (1,286 total)
  - **Test Set**: 750 Defective | 540 OK (1,290 total)

---

## 🛠️ How to Run the Web Application

### Prerequisites

Ensure Python 3.10+ is installed on your system.

### 1. Install Dependencies

Install the required Python libraries using the provided requirements file:

```bash
pip install -r requirements.txt
```

---

### 2. (Optional) Run Dataset Pipeline & Model Training

If you wish to re-run data auditing, training, or benchmark evaluation:

* **Audit and Clean Dataset**:
  ```bash
  python clean_dataset.py
  ```
* **Train All Models (YOLOv11, EfficientNet-B3, ViT-B/16)**:
  ```bash
  python train_all.py
  ```
* **Evaluate & Benchmark Models**:
  ```bash
  python evaluate_all.py
  ```

---

### 3. Launch the Application Server

Make sure your terminal is navigated into the correct directory (the folder containing `server.py`). If you extracted this project from a ZIP file, you might need to navigate into the nested folder first:
```bash
cd Indurstry-AI-main
```

Start the FastAPI production web server:

```bash
python server.py
```

The application will start on port `8000` with hot-reloading enabled.

---

### 4. Access the Web Dashboard

Open your web browser and navigate to:

👉 **[http://localhost:8000](http://localhost:8000)**

---

## 💻 Web Application Features & Navigation

1. **Dataset Analytics Tab**:
   - Live KPI cards (Clean Dataset Count, Defective/OK breakdown, Audit results).
   - Interactive split visualization charts and target model data mapping table.

2. **Model Matrix Tab**:
   - Comparative benchmark table showing Accuracy, Precision, Recall, F1-Score, Latency, and Parameters.
   - Interactive bar chart comparing inference latencies and model parameters.

3. **Live Inspection Tab**:
   - **Upload Custom Image**: Drag-and-drop or upload any casting sample.
   - **Pre-loaded Test Samples**: Click on pre-configured test cards from the test set.
   - **Multi-View Inspection Display**:
     - *YOLOv11 Bounding Box Overlay*: Exact defect bounding box location.
     - *EfficientNet Grad-CAM Heatmap*: Visual focus areas driving the pass/fail decision.
     - *ViT Patch Attention Map*: Self-attention weights highlight global structural flaws.
   - **Consensus Verdict & Diagnostic Report**: Final ensemble decision, confidence percentage, and model agreement rating.

---

## 📂 Project Directory Structure

```
Industry AI/
├── README.md                           # Project documentation & benchmark metrics
├── server.py                           # FastAPI application entry point
├── audit_dataset.py                    # Initial raw dataset audit script
├── clean_dataset.py                    # Perceptual hash deduplication & structure builder
├── train_all.py                        # Training pipeline for EfficientNet, ViT, & YOLOv11
├── evaluate_all.py                     # Evaluation script generating model_comparison.json
├── yolo11n.pt                          # Base pretrained YOLOv11 weight file
│
├── agent/
│   └── smart_manufacturing_agent.py    # Core Agent logic, consensus engine & XAI generator
│
├── models/
│   ├── efficientnet_model.py           # PyTorch EfficientNet-B3 architecture & Grad-CAM
│   ├── vit_model.py                    # PyTorch Vision Transformer architecture & Attention rollout
│   └── yolo_model.py                   # Ultralytics YOLOv11 wrapper & contour detector
│
├── static/
│   ├── index.html                      # Single-page web dashboard HTML
│   └── style.css                       # Modern dark glassmorphism design system
│
├── weights/                            # Trained model weights (.pth and .pt)
├── reports/                            # Generated benchmark metrics JSON files
└── dataset/                            # Raw and processed casting image dataset
```

---

## 📄 License & Acknowledgments

Developed for Industrial AI & Smart Manufacturing Defect Detection benchmarking. Built with PyTorch, Ultralytics YOLOv11, FastAPI, and Chart.js.
