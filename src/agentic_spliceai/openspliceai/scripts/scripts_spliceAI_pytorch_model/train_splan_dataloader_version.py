import argparse
import os, sys
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler, random_split
import platform
import spliceai
from splan_utils import *
from splan_constant import *
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
# import wandb

from torch.utils.data import Dataset
from utils import clip_datapoints, print_topl_statistics
from spliceai_dataset import *

RANDOM_SEED = 42

def setup_device():
    """Select computation device based on availability."""
    device_str = "cuda" if torch.cuda.is_available() else "mps" if platform.system() == "Darwin" else "cpu"
    return torch.device(device_str)


def initialize_paths(project_root, project_name, chunk_size, flanking_size, exp_num, training_target, sequence_length):
    """Initialize project directories and create them if they don't exist."""
    ####################################
    # Modify the model verson here!!
    ####################################
    MODEL_VERSION = f"{project_name}_{training_target}_{sequence_length}_{flanking_size}_{exp_num}"
    ####################################
    # Modify the model verson here!!
    ####################################
    model_train_outdir = f"{project_root}results/model_train_outdir/{MODEL_VERSION}/{exp_num}/"
    model_output_base = f"{model_train_outdir}models/"
    log_output_base = f"{model_train_outdir}LOG/"
    log_output_train_base = f"{log_output_base}TRAIN/"
    log_output_val_base = f"{log_output_base}VAL/"
    log_output_test_base = f"{log_output_base}TEST/"
    for path in [model_output_base, log_output_train_base, log_output_val_base, log_output_test_base]:
        os.makedirs(path, exist_ok=True)
    return model_output_base, log_output_train_base, log_output_val_base, log_output_test_base


def initialize_model_and_optim(device, flanking_size):
    """Initialize the model, criterion, optimizer, and scheduler."""
    # Hyper-parameters:
    # L: Number of convolution kernels
    # W: Convolution window size in each residual unit
    # AR: Atrous rate in each residual unit
    L = 32
    N_GPUS = 2
    W = np.asarray([11, 11, 11, 11])
    AR = np.asarray([1, 1, 1, 1])
    BATCH_SIZE = 18*N_GPUS
    if int(flanking_size) == 80:
        W = np.asarray([11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1])
        BATCH_SIZE = 18*N_GPUS
    elif int(flanking_size) == 400:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
        BATCH_SIZE = 18*N_GPUS
    elif int(flanking_size) == 2000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                        10, 10, 10, 10])
        BATCH_SIZE = 12*N_GPUS
    elif int(flanking_size) == 10000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21, 41, 41, 41, 41])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                        10, 10, 10, 10, 25, 25, 25, 25])
        BATCH_SIZE = 6*N_GPUS
    CL = 2 * np.sum(AR*(W-1))
    print("\033[1mContext nucleotides: %d\033[0m" % (CL))
    print("\033[1mSequence length (output): %d\033[0m" % (SL))
    model = spliceai.SpliceAI(L, W, AR).to(device)
    # criterion = nn.BCELoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, 1000, train_size * EPOCH_NUM)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6, 7, 8, 9], gamma=0.5)
    params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE}
    return model, criterion, optimizer, scheduler, params


def calculate_metrics(y_true, y_pred):
    """Calculate metrics including precision, recall, f1-score, and accuracy."""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy


def threshold_predictions(y_probs, threshold=0.5):
    """Threshold probabilities to get binary predictions."""
    return (y_probs > threshold).astype(int)


def valid_epoch(model, dataloader, criterion, device, params, metric_files, run_mode):
    print(f"\033[1m{run_mode.capitalize()}ing model...\033[0m")
    model.eval()
    running_loss = 0.0
    np.random.seed(RANDOM_SEED)  # You can choose any number as a seed
    batch_ylabel = []
    batch_ypred = []
    print_dict = {}
    for X_batch, Y_batch in dataloader:
        # print("\tBatch X shape:", X_batch.shape, file=sys.stderr)
        # print("\tBatch Y shape:", Y_batch.shape, file=sys.stderr)
        # for batch_idx, batch in enumerate(loader, 1):        
        DNAs = X_batch.to(device)
        labels = Y_batch.to(device)
        DNAs = DNAs.permute((0, 2, 1))
        labels = labels.permute((0, 2, 1))
        # print("\n\tDNAs.shape: ", DNAs.shape)
        # print("\tlabels.shape: ", labels.shape)
        DNAs, labels = clip_datapoints(DNAs, labels, params["CL"], 2)
        DNAs, labels = DNAs.to(torch.float32).to(device), labels.to(torch.float32).to(device)
        # print("\n\tAfter clipping DNAs.shape: ", DNAs.shape)
        # print("\tAfter clipping labels.shape: ", labels.shape)
        yp = model(DNAs)
        loss = categorical_crossentropy_2d(labels, yp)
        # loss = criterion(labels, yp)  # Permuting to [batch_size, sequence_length, num_classes]
        running_loss += loss.item()
        print("loss: ", loss.item())
        batch_ylabel.append(labels.detach().cpu())
        batch_ypred.append(yp.detach().cpu())
        print_dict["loss"] = loss.item()
    batch_ylabel = torch.cat(batch_ylabel, dim=0)
    batch_ypred = torch.cat(batch_ypred, dim=0)
    is_expr = (batch_ylabel.sum(axis=(1,2)) >= 1).cpu().numpy()
    if np.any(is_expr):
        ############################
        # Topk SpliceAI assessment approach
        ############################
        Y_true_1 = batch_ylabel[is_expr, 1, :].flatten().cpu().detach().numpy()
        Y_true_2 = batch_ylabel[is_expr, 2, :].flatten().cpu().detach().numpy()
        Y_pred_1 = batch_ypred[is_expr, 1, :].flatten().cpu().detach().numpy()
        Y_pred_2 = batch_ypred[is_expr, 2, :].flatten().cpu().detach().numpy()
        acceptor_topkl_accuracy, acceptor_auprc = print_topl_statistics(np.asarray(Y_true_1),
                            np.asarray(Y_pred_1), metric_files["topk_acceptor"], type='acceptor', print_top_k=True)
        donor_topkl_accuracy, donor_auprc = print_topl_statistics(np.asarray(Y_true_2),
                            np.asarray(Y_pred_2), metric_files["topk_donor"], type='donor', print_top_k=True)
        for k, v in metric_files.items():
            with open(v, 'a') as f:
                if k == "loss":
                    f.write(f"{running_loss / len(dataloader)}\n")
                elif k == "topk_acceptor":
                    f.write(f"{acceptor_topkl_accuracy}\n")
                elif k == "topk_donor":
                    f.write(f"{donor_topkl_accuracy}\n")
                elif k == "auprc_acceptor":
                    f.write(f"{acceptor_auprc}\n")
                elif k == "auprc_donor":
                    f.write(f"{donor_auprc}\n")
        # wandb.log({
        #     f'{run_mode}/loss': running_loss / len(dataloader),
        #     f'{run_mode}/topk_acceptor': acceptor_topkl_accuracy,
        #     f'{run_mode}/topk_donor': donor_topkl_accuracy,
        #     f'{run_mode}/auprc_acceptor': acceptor_auprc,
        #     f'{run_mode}/auprc_donor': donor_auprc,
        # })
        print("***************************************\n\n")


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, params, metric_files, run_mode):
    print(f"\033[1m{run_mode.capitalize()}ing model...\033[0m")
    model.train()
    running_loss = 0.0
    np.random.seed(RANDOM_SEED)  # You can choose any number as a seed
    batch_ylabel = []
    batch_ypred = []
    print_dict = {}
    counter = 0
    for X_batch, Y_batch in dataloader:
        # print("\tBatch X shape:", X_batch.shape, file=sys.stderr)
        # print("\tBatch Y shape:", Y_batch.shape, file=sys.stderr)
        # for batch_idx, batch in enumerate(loader, 1):        
        DNAs = X_batch.to(device)
        labels = Y_batch.to(device)
        DNAs = DNAs.permute((0, 2, 1))
        labels = labels.permute((0, 2, 1))
        # print("\n\tDNAs.shape: ", DNAs.shape)
        # print("\tlabels.shape: ", labels.shape)
        DNAs, labels = clip_datapoints(DNAs, labels, params["CL"], 2)
        DNAs, labels = DNAs.to(torch.float32).to(device), labels.to(torch.float32).to(device)
        # print("\n\tAfter clipping DNAs.shape: ", DNAs.shape)
        # print("\tAfter clipping labels.shape: ", labels.shape)
        optimizer.zero_grad()
        yp = model(DNAs)
        loss = categorical_crossentropy_2d(labels, yp)
        # loss = criterion(labels, yp)  # Permuting to [batch_size, sequence_length, num_classes]
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        batch_ylabel.append(labels.detach().cpu())
        batch_ypred.append(yp.detach().cpu())
        print_dict["loss"] = loss.item()
    batch_ylabel = torch.cat(batch_ylabel, dim=0)
    batch_ypred = torch.cat(batch_ypred, dim=0)
    is_expr = (batch_ylabel.sum(axis=(1,2)) >= 1).cpu().numpy()
    scheduler.step()
    if np.any(is_expr):
        ############################
        # Topk SpliceAI assessment approach
        ############################
        Y_true_1 = batch_ylabel[is_expr, 1, :].flatten().cpu().detach().numpy()
        Y_true_2 = batch_ylabel[is_expr, 2, :].flatten().cpu().detach().numpy()
        Y_pred_1 = batch_ypred[is_expr, 1, :].flatten().cpu().detach().numpy()
        Y_pred_2 = batch_ypred[is_expr, 2, :].flatten().cpu().detach().numpy()
        acceptor_topkl_accuracy, acceptor_auprc = print_topl_statistics(np.asarray(Y_true_1),
                            np.asarray(Y_pred_1), metric_files["topk_acceptor"], type='acceptor', print_top_k=True)
        donor_topkl_accuracy, donor_auprc = print_topl_statistics(np.asarray(Y_true_2),
                            np.asarray(Y_pred_2), metric_files["topk_donor"], type='donor', print_top_k=True)
        for k, v in metric_files.items():
            with open(v, 'a') as f:
                if k == "loss":
                    f.write(f"{running_loss/len(dataloader)}\n")
                elif k == "topk_acceptor":
                    f.write(f"{acceptor_topkl_accuracy}\n")
                elif k == "topk_donor":
                    f.write(f"{donor_topkl_accuracy}\n")
                elif k == "auprc_acceptor":
                    f.write(f"{acceptor_auprc}\n")
                elif k == "auprc_donor":
                    f.write(f"{donor_auprc}\n")
        # wandb.log({
        #     f'{run_mode}/loss': running_loss/len(dataloader),
        #     f'{run_mode}/topk_acceptor': acceptor_topkl_accuracy,
        #     f'{run_mode}/topk_donor': donor_topkl_accuracy,
        #     f'{run_mode}/auprc_acceptor': acceptor_auprc,
        #     f'{run_mode}/auprc_donor': donor_auprc,
        # })
        print("***************************************\n\n")



def main():
    # os.environ['WANDB_MODE'] = 'disabled'
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable-wandb', '-d', action='store_true', default=False)
    parser.add_argument('--project-root', '-p', type=str)
    parser.add_argument('--project-name', '-s', type=str)
    parser.add_argument('--flanking-size', '-f', type=int, default=80)
    parser.add_argument('--exp-num', '-e', type=str, default=0)
    parser.add_argument('--training-target', '-t', type=str, default="SpliceAI")
    parser.add_argument('--train-dataset', '-train', type=str)
    parser.add_argument('--test-dataset', '-test', type=str)
    parser.add_argument('--dataset-shuffle', '-shuffle', action='store_true', default=False)
    args = parser.parse_args()
    print("args: ", args, file=sys.stderr)
    project_root = args.project_root
    project_name = args.project_name
    sequence_length = 5000
    flanking_size = int(args.flanking_size)
    exp_num = args.exp_num
    training_target = args.training_target
    assert int(flanking_size) in [80, 400, 2000, 10000]
    assert training_target in ["MANE", "SpliceAI"]
    # if args.disable_wandb:
    #     os.environ['WANDB_MODE'] = 'disabled'
    # wandb.init(project=f'{project_name}_{training_target}_{sequence_length}_{flanking_size}_{exp_num}', reinit=True)
    device = setup_device()
    print("device: ", device, file=sys.stderr)
    model_output_base, log_output_train_base, log_output_val_base, log_output_test_base = initialize_paths(project_root, project_name, sequence_length, flanking_size, exp_num, training_target, sequence_length)
    print("* model_output_base: ", model_output_base, file=sys.stderr)
    print("* log_output_train_base: ", log_output_train_base, file=sys.stderr)
    print("* log_output_val_base: ", log_output_val_base, file=sys.stderr)
    print("* log_output_test_base: ", log_output_test_base, file=sys.stderr)
    training_dataset = args.train_dataset
    testing_dataset = args.test_dataset
    print("training_dataset: ", training_dataset, file=sys.stderr)
    print("testing_dataset: ", testing_dataset, file=sys.stderr)
    ##########################################
    # Load training and testing dataset
    ##########################################
    train_loaded_dataset = torch.load(training_dataset)
    # Assuming train_loaded_dataset is your dataset object
    total_train_samples = len(train_loaded_dataset)
    train_size = int(0.9 * total_train_samples)
    val_size = total_train_samples - train_size
    # Splitting the training dataset
    train_dataset, val_dataset = random_split(train_loaded_dataset, [train_size, val_size])
    test_dataset = torch.load(testing_dataset)
    print("train_dataset: ", len(train_dataset))
    print("val_dataset: ", len(val_dataset))
    print("test_dataset: ", len(test_dataset))

    ##########################################
    # Model initialization
    ##########################################
    model, criterion, optimizer, scheduler, params = initialize_model_and_optim(device, flanking_size)
    train_metric_files = {
        'topk_donor': f'{log_output_train_base}/donor_topk.txt',
        'auprc_donor': f'{log_output_train_base}/donor_accuracy.txt',
        'topk_acceptor': f'{log_output_train_base}/acceptor_topk.txt',
        'auprc_acceptor': f'{log_output_train_base}/acceptor_accuracy.txt',
        'loss': f'{log_output_train_base}/loss.txt'
    }
    valid_metric_files = {
        'topk_donor': f'{log_output_val_base}/donor_topk.txt',
        'auprc_donor': f'{log_output_val_base}/donor_accuracy.txt',
        'topk_acceptor': f'{log_output_val_base}/acceptor_topk.txt',
        'auprc_acceptor': f'{log_output_val_base}/acceptor_accuracy.txt',
        'loss': f'{log_output_val_base}/loss.txt'
    }
    test_metric_files = {
        'topk_donor': f'{log_output_test_base}/donor_topk.txt',
        'auprc_donor': f'{log_output_test_base}/donor_accuracy.txt',
        'topk_acceptor': f'{log_output_test_base}/acceptor_topk.txt',
        'auprc_acceptor': f'{log_output_test_base}/acceptor_accuracy.txt',
        'loss': f'{log_output_test_base}/loss.txt'
    }

    train_dataloader = DataLoader(train_dataset, batch_size=params['BATCH_SIZE'], shuffle=args.dataset_shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=params['BATCH_SIZE'], shuffle=args.dataset_shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=params['BATCH_SIZE'], shuffle=args.dataset_shuffle)

    print("train_dataloader: ", len(train_dataloader))
    print("test_dataloader: ", len(test_dataloader))
    
    for epoch in range(EPOCH_NUM):
        print("\n--------------------------------------------------------------")
        print(f">> Epoch {epoch + 1}")
        start_time = time.time()
        train_epoch(model, train_dataloader, criterion, optimizer, scheduler, device, params, train_metric_files, run_mode="train")
        valid_epoch(model, val_dataloader, criterion, device, params, valid_metric_files, run_mode="validation")
        valid_epoch(model, test_dataloader, criterion, device, params, test_metric_files, run_mode="test")
        torch.save(model.state_dict(), f"{model_output_base}/model_{epoch}.pt")
        print("--- %s seconds ---" % (time.time() - start_time))
        print("--------------------------------------------------------------")
        # if epoch == 1:
        #     break

if __name__ == "__main__":
    main()
