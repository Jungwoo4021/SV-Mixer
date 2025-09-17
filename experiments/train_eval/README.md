# 🏋️ Training SV-Mixer (Training Mode)

This directory provides scripts to **train SV-Mixer models** on large-scale speaker verification datasets (e.g., VoxCeleb2).  
The following guide shows how to train the **17-layer Large SV-Mixer** model from scratch.

---

## 📂 Dataset

The training scripts are designed for **VoxCeleb2** by default.  
Make sure you have the dataset prepared and the file paths set correctly before starting training.

---

## ▶ How to Run

1. **Download training code**

   Download the `./train_code` directory from this repository.  
   It contains the scripts to train SV-Mixer models.

2. **Edit arguments**  
   Open [`train_code/arguments.py`](./train_code/arguments.py) and update the following fields:

   - `train_list`: path to the VoxCeleb2 training list file ([📂 download](https://github.com/Jungwoo4021/experimental-resources/raw/main/train_samples/vox2_train_samples.txt))
   - `path_vox_O_trials`: path to the VoxCeleb1 test trial file ([📂 download](https://github.com/Jungwoo4021/experimental-resources/raw/main/test_trials/vox2_testO_trials.txt))
   - `path_musan`: path to the MUSAN noise training list file ([📂 download](https://github.com/Jungwoo4021/experimental-resources/raw/main/train_samples/musan.txt))
   - `path_rir`: path to the RIR reverberation training list file ([📂 download](https://github.com/Jungwoo4021/experimental-resources/raw/main/train_samples/rir_noises.txt))

3. **Run the training script**

   Example: training the **Large 17-layer model** on VoxCeleb2

   ```bash
   python train_code/main.py