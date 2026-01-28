import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import platform
from spliceai import *
from utils import *
from constants import *
import h5py
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from keras.models import load_model
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from itertools import cycle

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

RANDOM_SEED = 42

def setup_device():
    """Select computation device based on availability."""
    device_str = "cuda" if torch.cuda.is_available() else "mps" if platform.system() == "Darwin" else "cpu"
    return torch.device(device_str)


def initialize_paths(output_dir, project_name, flanking_size, sequence_length, data_type):
    """Initialize project directories and create them if they don't exist."""
    MODEL_VERSION = f"{project_name}_{sequence_length}_{flanking_size}"
    model_train_outdir = f"{output_dir}/{MODEL_VERSION}/"
    model_output_base = f"{model_train_outdir}models/"
    log_output_base = f"{model_train_outdir}LOG/"
    if data_type == "train":
        log_output_test_base = f"{log_output_base}TRAIN/"
    elif data_type == "test":
        log_output_test_base = f"{log_output_base}TEST/"
    for path in [model_output_base, log_output_test_base]:
        os.makedirs(path, exist_ok=True)
    return model_output_base, log_output_test_base


def load_data_from_shard(h5f, shard_idx, device, batch_size, params, shuffle=False):
    X = h5f[f'X{shard_idx}'][:].transpose(0, 2, 1)
    Y = h5f[f'Y{shard_idx}'][0, ...].transpose(0, 2, 1)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    ds = TensorDataset(X, Y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True, pin_memory=True)


def classwise_accuracy(true_classes, predicted_classes, num_classes):
    class_accuracies = []
    for i in range(num_classes):
        true_positives = np.sum((predicted_classes == i) & (true_classes == i))
        total_class_samples = np.sum(true_classes == i)
        if total_class_samples > 0:
            accuracy = true_positives / total_class_samples
        else:
            accuracy = 0.0
        class_accuracies.append(accuracy)
    return class_accuracies


def metrics(batch_ypred, batch_ylabel, metric_files):
    _, predicted_classes = torch.max(batch_ypred, 1)
    true_classes = torch.argmax(batch_ylabel, dim=1)
    true_classes = true_classes.numpy()
    predicted_classes = predicted_classes.numpy()
    true_classes_flat = true_classes.flatten()
    predicted_classes_flat = predicted_classes.flatten()
    accuracy = accuracy_score(true_classes_flat, predicted_classes_flat)
    precision, recall, f1, _ = precision_recall_fscore_support(true_classes_flat, predicted_classes_flat, average=None)
    class_accuracies = classwise_accuracy(true_classes, predicted_classes, 3)
    overall_accuracy = np.mean(class_accuracies)
    print(f"Overall Accuracy: {overall_accuracy}")
    for k, v in metric_files.items():
        with open(v, 'a') as f:
            if k == "accuracy":
                f.write(f"{overall_accuracy}\n")
    ss_types = ["Non-splice", "acceptor", "donor"]
    for i, (acc, prec, rec, f1_score) in enumerate(zip(class_accuracies, precision, recall, f1)):
        print(f"Class {ss_types[i]}\t: Accuracy={acc}, Precision={prec}, Recall={rec}, F1={f1_score}")
        for k, v in metric_files.items():
            with open(v, 'a') as f:
                if k == f"{ss_types[i]}_precision":
                    f.write(f"{prec}\n")
                elif k == f"{ss_types[i]}_recall":
                    f.write(f"{rec}\n")
                elif k == f"{ss_types[i]}_f1":
                    f.write(f"{f1_score}\n")
                elif k == f"{ss_types[i]}_accuracy":
                    f.write(f"{acc}\n")


def model_evaluation(batch_ylabel, batch_ypred, metric_files, run_mode):
    batch_ylabel = torch.cat(batch_ylabel, dim=0)
    batch_ypred = torch.cat(batch_ypred, dim=0)
    is_expr = (batch_ylabel.sum(axis=(1,2)) >= 1).cpu().numpy()
    if np.any(is_expr):
        subset_size = 1000
        indices = np.arange(batch_ylabel[is_expr].shape[0])
        subset_indices = np.random.choice(indices, size=min(subset_size, len(indices)), replace=False)
        Y_true_1 = batch_ylabel[is_expr][subset_indices, 1, :].flatten().cpu().detach().numpy()
        Y_true_2 = batch_ylabel[is_expr][subset_indices, 2, :].flatten().cpu().detach().numpy()
        Y_pred_1 = batch_ypred[is_expr][subset_indices, 1, :].flatten().cpu().detach().numpy()
        Y_pred_2 = batch_ypred[is_expr][subset_indices, 2, :].flatten().cpu().detach().numpy()
        plt.figure()
        acceptor_topk_accuracy, acceptor_auprc, acceptor_auroc = print_topl_statistics(np.asarray(Y_true_1),
                            np.asarray(Y_pred_1), metric_files, ss_type='acceptor', print_top_k=True)
        donor_topk_accuracy, donor_auprc, donor_auroc = print_topl_statistics(np.asarray(Y_true_2),
                            np.asarray(Y_pred_2), metric_files, ss_type='donor', print_top_k=True)
        plt.savefig(metric_files['prc'])
        plt.clf()
        loss = categorical_crossentropy_2d(batch_ylabel, batch_ypred)
        for k, v in metric_files.items():
            with open(v, 'a') as f:
                if k == "loss_batch":
                    f.write(f"{loss.item()}\n")
                elif k == "donor_topk":
                    f.write(f"{donor_topk_accuracy}\n")
                elif k == "donor_auprc":
                    f.write(f"{donor_auprc}\n")
                elif k == "donor_auroc":
                    f.write(f"{donor_auroc}\n")
                elif k == "acceptor_topk":
                    f.write(f"{acceptor_topk_accuracy}\n")
                elif k == "acceptor_auprc":
                    f.write(f"{acceptor_auprc}\n")
                elif k == "acceptor_auroc":
                    f.write(f"{acceptor_auroc}\n")
    metrics(batch_ypred, batch_ylabel, metric_files)
    batch_ylabel = []
    batch_ypred = []


def valid_epoch(model, h5f, idxs, batch_size, device, params, metric_files, run_mode, sample_freq):
    print(f"\033[1m{run_mode.capitalize()}ing model...\033[0m")
    model.eval()
    running_loss = 0.0
    np.random.seed(RANDOM_SEED)
    shuffled_idxs = np.random.choice(idxs, size=len(idxs), replace=False)
    shuffled_idxs = idxs[:30]
    batch_ylabel = []
    batch_ypred = []
    print_dict = {}
    batch_idx = 0
    for i, shard_idx in enumerate(shuffled_idxs, 1):
        print(f"Shard {i}/{len(shuffled_idxs)}")
        loader = load_data_from_shard(h5f, shard_idx, device, batch_size, params, shuffle=False)
        pbar = tqdm(loader, leave=False, total=len(loader), desc=f'Shard {i}/{len(shuffled_idxs)}')
        for batch in pbar:
            DNAs, labels = batch[0].to(device), batch[1].to(device)
            DNAs, labels = clip_datapoints(DNAs, labels, params["CL"], 2)
            DNAs, labels = DNAs.to(torch.float32).to(device), labels.to(torch.float32).to(device)
            yp = model(DNAs)
            loss = categorical_crossentropy_2d(labels, yp)
            with open(metric_files["loss_every_update"], 'a') as f:
                f.write(f"{loss.item()}\n")
            running_loss += loss.item()
            batch_ylabel.append(labels.detach().cpu())
            batch_ypred.append(yp.detach().cpu())
            print_dict["loss"] = loss.item()
            pbar.set_postfix(print_dict)
            pbar.update(1)
            batch_idx += 1
        pbar.close()
    model_evaluation(batch_ylabel, batch_ypred, metric_files, run_mode)


def initialize_model_and_optim(flanking_size):
    L = 32
    N_GPUS = 2
    W = np.asarray([11, 11, 11, 11])
    AR = np.asarray([1, 1, 1, 1])
    BATCH_SIZE = 18 * N_GPUS
    if int(flanking_size) == 80:
        W = np.asarray([11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1])
        BATCH_SIZE = 18 * N_GPUS
    elif int(flanking_size) == 400:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
        BATCH_SIZE = 18 * N_GPUS
    elif int(flanking_size) == 2000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                        10, 10, 10, 10])
        BATCH_SIZE = 12 * N_GPUS
    elif int(flanking_size) == 10000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21, 41, 41, 41, 41])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                        10, 10, 10, 10, 25, 25, 25, 25])
        BATCH_SIZE = 6 * N_GPUS
    CL = 2 * np.sum(AR * (W - 1))
    params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE}
    return params


def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', '-p', type=str)
    parser.add_argument('--flanking-size', '-f', type=int, default=80)
    parser.add_argument('--test-dataset', '-test', type=str)
    parser.add_argument('--project-name', '-s', type=str)
    parser.add_argument('--model', '-m', default="SpliceAI", type=str)
    parser.add_argument('--type', '-t', default="test", type=str, help='train or test')
    args = parser.parse_args()
    output_dir = args.output_dir
    project_name = args.project_name
    sequence_length = 5000
    flanking_size = int(args.flanking_size)
    model_path = args.model
    testing_dataset = args.test_dataset
    assert int(flanking_size) in [80, 400, 2000, 10000]
    device = setup_device()
    model_output_base, log_output_test_base = initialize_paths(output_dir, project_name, flanking_size, sequence_length, args.type)
    test_h5f = h5py.File(testing_dataset, 'r')
    np.random.seed(RANDOM_SEED)
    test_idxs = np.arange(len(test_h5f.keys()) // 2)
    params = initialize_model_and_optim(flanking_size)
    model = SpliceAI(params['L'], params['W'], params['AR'])
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    test_metric_files = {
        'donor_topk_all': f'{log_output_test_base}/donor_topk_all.txt',
        'donor_topk': f'{log_output_test_base}/donor_topk.txt',
        'donor_auprc': f'{log_output_test_base}/donor_auprc.txt',
        'donor_auroc': f'{log_output_test_base}/donor_auroc.txt',
        'donor_accuracy': f'{log_output_test_base}/donor_accuracy.txt',
        'donor_precision': f'{log_output_test_base}/donor_precision.txt',
        'donor_recall': f'{log_output_test_base}/donor_recall.txt',
        'donor_f1': f'{log_output_test_base}/donor_f1.txt',
        'acceptor_topk_all': f'{log_output_test_base}/acceptor_topk_all.txt',
        'acceptor_topk': f'{log_output_test_base}/acceptor_topk.txt',
        'acceptor_auprc': f'{log_output_test_base}/acceptor_auprc.txt',
        'acceptor_auroc': f'{log_output_test_base}/acceptor_auroc.txt',
        'acceptor_accuracy': f'{log_output_test_base}/acceptor_accuracy.txt',
        'acceptor_precision': f'{log_output_test_base}/acceptor_precision.txt',
        'acceptor_recall': f'{log_output_test_base}/acceptor_recall.txt',
        'acceptor_f1': f'{log_output_test_base}/acceptor_f1.txt',
        'prc': f'{log_output_test_base}/prc.png',
        'accuracy': f'{log_output_test_base}/accuracy.txt',
        'loss_batch': f'{log_output_test_base}/loss_batch.txt',
        'loss_every_update': f'{log_output_test_base}/loss_every_update.txt',
    }
    SAMPLE_FREQ = 1000
    print("\n--------------------------------------------------------------")
    start_time = time.time()
    BATCH_SIZE = 36
    valid_epoch(model, test_h5f, test_idxs, BATCH_SIZE, device, params, test_metric_files, run_mode="test", sample_freq=SAMPLE_FREQ)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("--------------------------------------------------------------")
    test_h5f.close()

if __name__ == "__main__":
    predict()
