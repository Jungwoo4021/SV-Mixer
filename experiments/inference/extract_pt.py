import torch
from collections import OrderedDict

# Lightning ckpt 경로
ckpt_path = "svmixer_5layer_eer0.91.ckpt"
ckpt = torch.load(ckpt_path, map_location="cpu")

# Lightning ckpt는 보통 {"state_dict": ..., ...} 형태
state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

def extract_submodule(sd, prefix):
    """
    특정 prefix(예: 'student_model.')로 시작하는 파라미터만 추출
    """
    new_sd = OrderedDict()
    for k, v in sd.items():
        if k.startswith(prefix):
            new_sd[k[len(prefix):]] = v  # prefix 제거
    return new_sd

# 1. student_model 가중치
student_sd = extract_submodule(state_dict, "student_model.")
torch.save(student_sd, "student_model.pt")
print("✅ Saved student_model.pt")

# 2. classifier 가중치
classifier_sd = extract_submodule(state_dict, "classifier.")
torch.save(classifier_sd, "classifier.pt")
print("✅ Saved classifier.pt")
