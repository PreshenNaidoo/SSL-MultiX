# Consensus-Guided Evaluation of Self-Supervised Learning in Echocardiographic Segmentation

The complete source code and dataset will be released with the research paper.


## Downloading the Datasets

The following datasets are used in this research study:

1. **Unity Dataset** – Available at [data.unityimaging.net](https://data.unityimaging.net/)  
   - Used for training and fine-tuning.  

2. **UnityLV-MultiX Dataset** – Introduced in this study, available soon at [intsav.com/unitylv-multix](https://www.intsav.com/unitylv-multix)  
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


   
