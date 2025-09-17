from dataclasses import dataclass
from collections import defaultdict

@dataclass
class EnrollmentItem:
    key: int
    path: str

@dataclass
class Trial:
    key1: int
    key2: int
    label: int

class VCMixTest:
    def __init__(self, test_trials):
        self.enrollment_set = []
        self.trials = []
        seen_keys = defaultdict(int)
        with open(test_trials, 'r') as f:
            for line in f:
                label, key1, key2, sample1, sample2 = line.strip().split(' ')
                label, key1, key2 = int(label), int(key1), int(key2)
                
                item = Trial(key1, key2, label)
                self.trials.append(item)

                for key, sample in [(key1, sample1), (key2, sample2)]:
                    if seen_keys[key] == 0:
                        self.enrollment_set.append(EnrollmentItem(key, sample))
                        seen_keys[key] = 1