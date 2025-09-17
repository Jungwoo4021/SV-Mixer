# hubconf.py

import torch
import torch.nn as nn

# 필수: 이 레포에서 제공하는 엔트리포인트 함수 목록
dependencies = ["torch", "torchvision"]

def small_svmixer(pretrained=False, **kwargs):
    from experiments.eval_only.test_code.models.svmixer import SVMixer

    model = SVMixer(5, 149, 1024)

    if pretrained:
        # 사전 학습된 가중치를 불러오기 (예: 릴리스에 올려둔 파일에서 다운로드)
        ckpt = torch.hub.load_state_dict_from_url(
            "https://github.com/Jungwoo4021/SV-Mixer/raw/main/assets/trained_models/svmixer_5layer_eer0.91_student_model.pt",
            map_location="cpu"
        )
        model.load_state_dict(ckpt)
    return model

def large_svmixer(pretrained=False, **kwargs):
    from experiments.eval_only.test_code.models.svmixer import SVMixer

    model = SVMixer(17, 149, 1024)

    if pretrained:
        # 사전 학습된 가중치를 불러오기 (예: 릴리스에 올려둔 파일에서 다운로드)
        ckpt = torch.hub.load_state_dict_from_url(
            "https://github.com/Jungwoo4021/SV-Mixer/raw/main/assets/trained_models/svmixer_17layer_eer0.78_student_model.pt",
            map_location="cpu"
        )
        model.load_state_dict(ckpt)
    return model

def small_classifier(pretrained=False, **kwargs):
    from experiments.eval_only.test_code.models.ecapa import ECAPA_TDNN

    model = ECAPA_TDNN(5, 1024, 512, 192)

    if pretrained:
        # 사전 학습된 가중치를 불러오기 (예: 릴리스에 올려둔 파일에서 다운로드)
        ckpt = torch.hub.load_state_dict_from_url(
            "https://github.com/Jungwoo4021/SV-Mixer/raw/main/assets/trained_models/svmixer_5layer_eer0.91_classifier.pt",
            map_location="cpu"
        )
        model.load_state_dict(ckpt)
    return model

def large_classifier(pretrained=False, **kwargs):
    from experiments.eval_only.test_code.models.ecapa import ECAPA_TDNN

    model = ECAPA_TDNN(17, 1024, 512, 192)

    if pretrained:
        # 사전 학습된 가중치를 불러오기 (예: 릴리스에 올려둔 파일에서 다운로드)
        ckpt = torch.hub.load_state_dict_from_url(
            "https://github.com/Jungwoo4021/SV-Mixer/raw/main/assets/trained_models/svmixer_17layer_eer0.78_classifier.pt",
            map_location="cpu"
        )
        model.load_state_dict(ckpt)
    return model