# 🔊 SV-Mixer Inference Demo (Colab)

This example demonstrates how to use the pretrained **Small SV-Mixer** and **ECAPA-TDNN Classifier** from [SV-Mixer](https://github.com/Jungwoo4021/SV-Mixer) to extract **hidden states** and **speaker embeddings** from a single audio sample.

---

## 🚀 Open Colab
Go to [Google Colab](https://colab.research.google.com/drive/1_zGof1NGM5WgZ5sJtQfsy1D7rKq9RwxR?usp=sharing) and see the 1-line example.

---

## ⚡ Quick Usage with Your Scripts
If you only want to use the pretrained models, you can easily load them with `torch.hub`:

```python
import torch

model = torch.hub.load("Jungwoo4021/SV-Mixer", "large", pretrained=True)
embedding = model(input_wav)
```

## 🔧 Model Selection

Two pretrained SV-Mixer variants are provided:

- **Small model**
  - Torch hub KEY: `small_svmixer`, `small_classifier`
  - 5 layers  
  - Size: 33.0M parameters  
  - GMACs: 11.9  
  - Vox1-O EER: 0.91%

- **Large model**  
  - Torch hub KEY: `large_svmixer`, `large_classifier`
  - 17 layers  
  - Size: 80.0M  
  - GMACs: 19.4  
  - Vox1-O EER: 0.78%