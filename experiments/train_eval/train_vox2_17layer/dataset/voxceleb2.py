import os
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class TrainItem:
    path: str
    speaker: str
    label: int

@dataclass
class EnrollmentItem:
    key: int
    path: str

@dataclass
class TestTrial:
    key1: int
    key2: int
    label: int

class VoxCeleb2:
    def __init__(self, train_samples, test_trials_O, test_trials_E=None, test_trials_H=None):
        # train_set
        self.train_set = []
        class_counts = defaultdict(int)
        with open(train_samples, 'r') as f:
            for line in f:
                line = line.strip()
                spk, label, wav_path = line.split()
                
                # convert spk(str) -> label(int)
                label = int(label)
                class_counts[label] += 1

                item = TrainItem(wav_path, spk, label)
                self.train_set.append(item)
        
        # class weight (inverse frequency)
        total_samples = len(self.train_set)
        num_classes = len(class_counts)
        self.class_weight = []

        for class_idx in range(num_classes):
            count = class_counts[class_idx]
            weight = total_samples / count
            self.class_weight.append(weight)

        # test_trials
        self.trials_O, self.enrollment_set_O = self.parse_trials(test_trials_O)

        if test_trials_E is not None:
            self.trials_E, self.enrollment_set_E = self.parse_trials(test_trials_E)

        if test_trials_H is not None:
            self.trials_H, self.enrollment_set_H = self.parse_trials(test_trials_H)

    def parse_trials(self, path):
        trials = []
        samples = []
        seen_keys = defaultdict(int)
        with open(path, 'r') as f:
            for line in f:
                label, key1, key2, sample1, sample2 = line.strip().split(' ')
                label, key1, key2 = int(label), int(key1), int(key2)
                
                item = TestTrial(key1, key2, label)
                trials.append(item)

                for key, sample in [(key1, sample1), (key2, sample2)]:
                    if seen_keys[key] == 0:
                        samples.append(EnrollmentItem(key, sample))
                        seen_keys[key] = 1

        return trials, samples