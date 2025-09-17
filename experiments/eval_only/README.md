# 🧪 In-Depth Evaluation (Evaluation-Only Mode)

This directory provides scripts to **evaluate pretrained SV-Mixer models** on multiple benchmark datasets without training.

## 📊 Supported Benchmarks

Running the evaluation will automatically test on:

- **VoxCeleb1-O**
- **VoxCeleb1-E**
- **VoxCeleb1-H**
- **VOiCES**
- **VoxSRC2023**
- **VCMix**

## 🔧 Model Selection

Two pretrained SV-Mixer variants are provided:

- **Small model**  
  - 5 layers  
  - Size: 33.0M parameters  
  - GMACs: 11.9  
  - Vox1-O EER: 0.91%

- **Large model**  
  - 17 layers  
  - Size: 80.0M  
  - GMACs: 19.4  
  - Vox1-O EER: 0.78%

## ▶ How to Run

1. **Download**

   Download the `./test_code` scripts from this repository and make sure they are available in your local environment.
   These scripts contain the evaluation pipeline for all supported benchmarks.

2. **Edit arguments**  
   Open [`test_code/arguments.py`](./test_code/arguments.py) and update the following fields:

   - `trial_file`: path to the downloaded evaluation trial file (see [📂 Trial Files](https://github.com/Jungwoo4021/experimental-resources/tree/main/test_trials))
     - vox2_testO_trials.txt  
     - vox2_testE_trials.txt  
     - vox2_testH_trials.txt  
     - vcmix_test.txt  
     - voxsrc_test.txt  
     - voices_eval.txt  

   - `model_type`: `'Small'` or `'Large'`

3. **Run the script**

   ```bash
   python test_code/main.py
