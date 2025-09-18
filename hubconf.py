# hubconf.py

import torch
import torch.nn as nn

from experiments.eval_only.test_code.models.svmixer import SVMixer
from experiments.eval_only.test_code.models.ecapa import ECAPA_TDNN

# 필수: 이 레포에서 제공하는 엔트리포인트 함수 목록
dependencies = ["torch", "torchvision"]

class ModelWrapper(nn.Module):
    def __init__(self, sv_mixer, classifier) -> None:
        super().__init__()

        self.sv_mixer = sv_mixer
        self.classifier = classifier

    def forward(self, x):
        x = self.sv_mixer(x)
        x = self.classifier(x)
        return x

def small(pretrained=False, **kwargs):
    """
    5 layers
    Size: 33.0M parameters
    GMACs: 11.9
    Vox1-O EER: 0.91%
    """
    sv_mixer = SVMixer(5, 149, 1024)
    ecapa = ECAPA_TDNN(5, 1024, 512, 192)

    if pretrained:
        # 사전 학습된 가중치를 불러오기 (예: 릴리스에 올려둔 파일에서 다운로드)
        sv_mixer_ckpt = torch.hub.load_state_dict_from_url(
            "https://github.com/Jungwoo4021/SV-Mixer/raw/main/assets/trained_models/svmixer_5layer_eer0.91_student_model.pt",
            map_location="cpu"
        )
        ecapa_ckpt = torch.hub.load_state_dict_from_url(
            "https://github.com/Jungwoo4021/SV-Mixer/raw/main/assets/trained_models/svmixer_5layer_eer0.91_classifier.pt",
            map_location="cpu"
        )
        sv_mixer.load_state_dict(sv_mixer_ckpt)
        ecapa.load_state_dict(ecapa_ckpt)

    model = ModelWrapper(sv_mixer, ecapa)
    
    return model

def large(pretrained=False, **kwargs):
    """
    17 layers
    Size: 80.0M
    GMACs: 19.4
    Vox1-O EER: 0.78%
    """
    sv_mixer = SVMixer(17, 149, 1024)
    ecapa = ECAPA_TDNN(17, 1024, 512, 192)

    if pretrained:
        # 사전 학습된 가중치를 불러오기 (예: 릴리스에 올려둔 파일에서 다운로드)
        sv_mixer_ckpt = torch.hub.load_state_dict_from_url(
            "https://github.com/Jungwoo4021/SV-Mixer/raw/main/assets/trained_models/svmixer_17layer_eer0.78_student_model.pt",
            map_location="cpu"
        )
        ecapa_ckpt = torch.hub.load_state_dict_from_url(
            "https://github.com/Jungwoo4021/SV-Mixer/raw/main/assets/trained_models/svmixer_17layer_eer0.78_classifier.pt",
            map_location="cpu"
        )
        sv_mixer.load_state_dict(sv_mixer_ckpt)
        ecapa.load_state_dict(ecapa_ckpt)

    model = ModelWrapper(sv_mixer, ecapa)
    
    return model