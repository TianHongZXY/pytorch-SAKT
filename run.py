import time
import torch
import numpy as np
import torch.nn as nn
from dataset import DataPrefetcher
from config import DefaultConfig
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import metrics
from sklearn.metrics import r2_score

opt = DefaultConfig()

def run_epoch(m, dataloader, optimizer, scheduler, criterion, num_skills,
                  epoch_id=None, writer=None, is_training=True):
    epoch_start_time = time.time()
    if is_training:
        m.train()
    else:
        m.eval()
    m.cuda()
    actual_labels = []
    pred_labels = []
    num_batch = len(dataloader)
    prefetcher = DataPrefetcher(dataloader, device='cuda')
    batch = prefetcher.next()
    k = 0

    if is_training:
        while batch is not None:
            target_index = []
            x, problems, correctness = batch
            x = x.long()
            problems = problems.long()
            correctness = correctness.view(-1).float()

            actual_labels += list(np.array(correctness))
            offset = 0
            helper = np.array(problems.cpu()).reshape(-1)
            for i in range(problems.size(0)):
                for j in range(problems.size(1)):
                    target_index.append((offset + helper[i * problems.size(1) + j + 1]))
                    offset += num_skills
            logits = m(x, problems, target_index, correctness)
            pred = torch.sigmoid(logits)
            loss = criterion(pred, correctness.cuda())
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), opt.max_grad_norm)
            optimizer.step()
            scheduler.step()
            pred_labels += list(np.array(pred.data.cpu()))
            batch = prefetcher.next()
            k += 1
            if k % 500 == 0:
                print('\r batch{}/{}'.format(k, num_batch), end='')
            if k >= num_batch - 1:
                break
    else:
        with torch.no_grad():
            while batch is not None:
                target_index = []
                x, problems, correctness = batch
                x = x.long()
                actual_num_problems = torch.sum(problems != num_skills, dim=1)
                num_problems = problems.size(1)
                problems = problems.long()
                correctness = correctness.view(-1).float()
                offset = 0
                helper = np.array(problems.cpu()).reshape(-1)
                for i in range(problems.size(0)):
                    for j in range(problems.size(1)):
                        target_index.append((offset + helper[i * problems.size(1) + j]))
                        offset += num_skills

                logits = m(x, problems, target_index, correctness)
                pred = torch.sigmoid(logits)
                for J in range(x.size(0)):
                    actual_num_problem = actual_num_problems[J]
                    num_to_throw = num_problems - actual_num_problem

                    pred[J * num_problems:J * num_problems + num_to_throw] = correctness[
                                                                             J * num_problems:J * num_problems + num_to_throw]
                actual_labels += list(np.array(correctness))
                pred_labels += list(np.array(pred.data.cpu()))
                batch = prefetcher.next()
                k += 1
                if k % 500 == 0:
                    print('\r batch{}/{}'.format(k, num_batch), end='')
                if k >= num_batch - 1:
                    break

    rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
    fpr, tpr, thresholds = metrics.roc_curve(actual_labels, pred_labels, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    r2 = r2_score(actual_labels, pred_labels)
    acc = metrics.accuracy_score(actual_labels, np.array(pred_labels) >= 0.5)
    epoch_end_time = time.time()
    print('Epoch costs %.2f s' % (epoch_end_time - epoch_start_time))
    return rmse, auc, r2, acc