## 📄 Publication

This work has been published as:

**Consensus-Guided Evaluation of Self-Supervised Learning in Echocardiographic Segmentation**  
Preshen Naidoo, Patricia Fernandes, Nasim Dadashi Serej, Charlotte H. Manisty, Matthew J. Shun-Shin, Darrel P. Francis, Massoud Zolgharni  
*Computers in Biology and Medicine*, Volume 198 (Nov 2025), Article 111148  
📄 https://doi.org/10.1016/j.compbiomed.2025.111148

### Abstract

We investigate self-supervised learning (SSL) for left ventricle segmentation in echocardiography, comparing multiple pretext tasks. We introduce the *UnityLV-MultiX* dataset with multi-expert consensus labels and propose a consensus-guided evaluation protocol to improve reliability. Our results show that contrastive SSL methods, when fine-tuned with limited labeled data (≈15%), achieve stronger alignment with multi-expert consensus than individual expert annotations — highlighting the promise of SSL to reduce labeling burden while improving model robustness and reproducibility.


## Downloading the Datasets

The following datasets are used in this research study:

1. **Unity Dataset** – Available at [data.unityimaging.net](https://data.unityimaging.net/)  
   - Used for training and fine-tuning.  

2. **UnityLV-MultiX Dataset** – Introduced in this study, available at [thrive-centre.com/unitylv-multix](https://www.thrive-centre.com/datasets/UnityLV-MultiX)  
   - Contains multi-expert labels and is used for evaluation.  


## Packages

The following packages are required:

- **Python** 3.10  
- **TensorFlow** 2.15.0  
- **TensorFlow-Addons** 0.22.0  
- **Keras** 2.15.0  
- **Pandas** 1.5.3  
- **NumPy** 1.24.4  
- **OpenCV** 4.5.5.64


## Pretext Tasks and Execution

Each pretext task in this study is organized into separate source files, allowing for easy configuration of hyperparameters.  

The following scripts can be executed independently. After running **experiments 1-6**, specify their output paths in `evaluate.py` to generate the combined results.  

### Pretext Task Scripts:
1. **`run_baseline.py`** – Establishes a randomly initialized baseline without pre-training.  
2. **`run_ssl_btwin.py`** – Implements the Barlow Twins pretext task.  
3. **`run_ssl_patch_rand.py`** – Pretext tasks for region-based and strip-based masking strategies.  
4. **`run_ssl_rot_echo.py`** – Random rotation pretext task.  
5. **`run_ssl_simclr.py`** – Implements the SimCLR pretext task.  
6. **`run_ssl_split.py`** – Combines contrastive learning and inpainting strategies.  
7. **`evaluate.py`** – Aggregates and analyzes results from the above experiments.  


   
