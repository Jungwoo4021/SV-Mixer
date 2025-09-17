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

class VOiCES:
    def __init__(self, dev_trials, eval_trials):
        self.dev_enrollment_set, self.dev_trials = self.read_file(dev_trials)
        self.eval_enrollment_set, self.eval_trials = self.read_file(eval_trials)

    def read_file(self, path):
        enrollment_set = []
        trials = []
        seen_keys = defaultdict(int)
        with open(path, 'r') as f:
            for line in f:
                label, key1, key2, sample1, sample2 = line.strip().split(' ')
                label, key1, key2 = int(label), int(key1), int(key2)
                
                item = Trial(key1, key2, label)
                trials.append(item)

                for key, sample in [(key1, sample1), (key2, sample2)]:
                    if seen_keys[key] == 0:
                        enrollment_set.append(EnrollmentItem(key, sample))
                        seen_keys[key] = 1

        return enrollment_set, trials