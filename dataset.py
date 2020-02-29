import csv
import torch
import time
import itertools
import numpy as np
from config import DefaultConfig
from wordtest import WordTestResource
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

opt = DefaultConfig()

class Data(Dataset):
    def __init__(self, train=True):
        start_time = time.time()
        if train:
            fileName = opt.train_data
        else:
            fileName = opt.test_data
        self.students = []
        self.max_skill_num = 0
        begin_index = 1e9
        with open(fileName, "r") as csvfile:
            for num_ques, ques, ans in itertools.zip_longest(*[csvfile] * 3):
                num_ques = int(num_ques.strip().strip(','))
                ques = [int(q) for q in ques.strip().strip(',').split(',')]
                ans = [int(a) for a in ans.strip().strip(',').split(',')]
                tmp_max_skill = max(ques)
                tmp_min_skill = min(ques)
                begin_index = min(tmp_min_skill, begin_index)
                self.max_skill_num = max(tmp_max_skill, self.max_skill_num)

                if (num_ques <= 2):
                    continue
                elif num_ques <= opt.max_len:
                    problems = np.zeros(opt.max_len, dtype=np.int64)
                    correct = np.ones(opt.max_len, dtype=np.int64)
                    problems[-num_ques:] = ques[-num_ques:]
                    correct[-num_ques:] = ans[-num_ques:]
                    self.students.append((num_ques, problems, correct))
                else:
                    start_idx = 0
                    while opt.max_len + start_idx <= num_ques:
                        problems = np.array(ques[start_idx:opt.max_len + start_idx])
                        correct = np.array(ans[start_idx:opt.max_len + start_idx])
                        tup = (opt.max_len, problems, correct)
                        start_idx += opt.max_len
                        self.students.append(tup)
                    left_num_ques = num_ques - start_idx
                    problems = np.zeros(opt.max_len, dtype=np.int64)
                    correct = np.ones(opt.max_len, dtype=np.int64)
                    problems[-left_num_ques:] = ques[start_idx:]
                    correct[-left_num_ques:] = ans[start_idx:]
                    tup = (left_num_ques, problems, correct)
                    self.students.append(tup)

    def __getitem__(self, index):
        student = self.students[index]
        problems = student[1]
        correct = student[2]
        x = np.zeros(opt.max_len - 1)
        x = problems[:-1]
        # we assume max_skill_num + 1 = num_skills because skill index starts from 0 to max_skill_num
        x += (correct[:-1] == 1) * (self.max_skill_num + 1)
        problems = problems[1:]
        correct = correct[1:]
        return x, problems, correct

    def __len__(self):
        return len(self.students)


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class DataPrefetcher():
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.device, non_blocking=True)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch