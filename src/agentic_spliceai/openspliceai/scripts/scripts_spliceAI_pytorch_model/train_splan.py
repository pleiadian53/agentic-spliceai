import argparse
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import platform
import spliceai
import spliceai_multihead
import DNATransformerConformer
import DNALocalTransformer
import DNALocalTransformerEncDec
import LocalTransformer
from splan_utils import *
from splan_constant import *
from tqdm import tqdm
import h5py
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import wandb
from utils import clip_datapoints, print_topl_statistics

RANDOM_SEED = 42

def setup_device():
    """Select computation device based on availability."""
    device_str = "cuda" if torch.cuda.is_available() else "mps" if platform.system() == "Darwin" else "cpu"
    return torch.device(device_str)


def initialize_paths(project_root, project_name, chunk_size, flanking_size, exp_num, training_target, sequence_length, model_arch):
    """Initialize project directories and create them if they don't exist."""
    ####################################
    # Modify the model verson here!!
    ####################################
    MODEL_VERSION = f"{model_arch}_{project_name}_{training_target}_{sequence_length}_{flanking_size}_{exp_num}"
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


def initialize_model_and_optim(device, flanking_size, model_arch):
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
    if model_arch == "SpliceAI":
        model = spliceai.SpliceAI(L, W, AR).to(device)
    elif model_arch == "LocalTransformer":
        BATCH_SIZE = 8
        CL = 0
        sequence_len = 5000
        dna_dimension = 4
        output_dimension = 3
        model = LocalTransformer.LocalTransformer(
            num_tokens = output_dimension,
            dim = 512,
            depth = 6,
            max_seq_len = sequence_len,
            causal = True,
            local_attn_window_size = int(flanking_size)
        ).to(device)
    elif model_arch == "SpliceAI_Multihead":
        BATCH_SIZE = 8
        CL = 0
        model = spliceai_multihead.SpliceAI_Multihead(L, W, AR).to(device)
    elif model_arch == "DNATransformerConformer":
        # Setting batch size to 10 for transformer-based model
        BATCH_SIZE = 8
        CL = 0
        # Model instantiation example
        ninp = 512 # embedding dimension
        nhead = 8 # number of attention heads
        nhid = 2048 # dimension of the feedforward network model in nn.TransformerEncoder
        nconformers = 6 # number of nn.TransformerEncoderLayer
        dropout = 0.1 # dropout rate
        kernel_size = 33  # Kernel size for the convolutional layers in the Conformer blocks
        model = DNATransformerConformer.DNATransformerConformer(ninp=ninp, nhead=nhead, nhid=nhid, nconformers=nconformers, dropout=dropout, kernel_size=kernel_size).to(device)
    elif model_arch == "DNALocalTransformer":
        # Model parameters setup
        BATCH_SIZE = 8
        CL = 0
        embed_size = 512
        num_layers = 2
        heads = 8
        forward_expansion = 4
        dropout = 0.1
        # Example model instantiation
        model = DNALocalTransformer.DNALocalTransformer(embed_size, num_layers, heads, device, forward_expansion, dropout).to(device)
    elif model_arch == "DNALocalTransformerEncDec":
        # Model parameters setup
        BATCH_SIZE = 8
        CL = 0
        embed_size = 512
        num_layers = 2
        heads = 8
        forward_expansion = 4
        dropout = 0.1
        # Example model instantiation
        model = DNALocalTransformer.DNALocalTransformer(embed_size, num_layers, heads, device, forward_expansion, dropout).to(device)
    print(model, file=sys.stderr)
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


def load_data_from_shard(h5f, shard_idx, device, batch_size, params, shuffle=False):
    X = h5f[f'X{shard_idx}'][:].transpose(0, 2, 1)
    Y = h5f[f'Y{shard_idx}'][0, ...].transpose(0, 2, 1)
    # print("\n\tX.shape: ", X.shape)
    # print("\tY.shape: ", Y.shape)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    ds = TensorDataset(X, Y)
    # print("\rds: ", ds)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True, pin_memory=True)
    # return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=8, pin_memory=True)


def valid_epoch(model, h5f, idxs, batch_size, criterion, device, params, metric_files, run_mode):
    print(f"\033[1m{run_mode.capitalize()}ing model...\033[0m")
    model.eval()
    running_loss = 0.0
    np.random.seed(RANDOM_SEED)  # You can choose any number as a seed
    shuffled_idxs = np.random.choice(idxs, size=len(idxs), replace=False)    
    print("shuffled_idxs: ", shuffled_idxs)
    batch_ylabel = []
    batch_ypred = []
    print_dict = {}
    for i, shard_idx in enumerate(shuffled_idxs, 1):
        print(f"Shard {i}/{len(shuffled_idxs)}")
        loader = load_data_from_shard(h5f, shard_idx, device, batch_size, params, shuffle=False)
        pbar = tqdm(loader, leave=False, total=len(loader), desc=f'Shard {i}/{len(shuffled_idxs)}')
        for batch in pbar:
            DNAs, labels = batch[0].to(device), batch[1].to(device)
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
            # print("loss: ", loss.item())
            batch_ylabel.append(labels.detach().cpu())
            batch_ypred.append(yp.detach().cpu())
            print_dict["loss"] = loss.item()
            pbar.set_postfix(print_dict)
            pbar.update(1)           
        pbar.close()
    batch_ylabel = torch.cat(batch_ylabel, dim=0)
    batch_ypred = torch.cat(batch_ypred, dim=0)
    is_expr = (batch_ylabel.sum(axis=(1,2)) >= 1).cpu().numpy()
    if np.any(is_expr):
        ############################
        # Topk SpliceAI assessment approach
        ############################
        subset_size = 1000
        indices = np.arange(batch_ylabel[is_expr].shape[0])
        subset_indices = np.random.choice(indices, size=min(subset_size, len(indices)), replace=False)
        Y_true_1 = batch_ylabel[is_expr][subset_indices, 1, :].flatten().cpu().detach().numpy()
        Y_true_2 = batch_ylabel[is_expr][subset_indices, 2, :].flatten().cpu().detach().numpy()
        Y_pred_1 = batch_ypred[is_expr][subset_indices, 1, :].flatten().cpu().detach().numpy()
        Y_pred_2 = batch_ypred[is_expr][subset_indices, 2, :].flatten().cpu().detach().numpy()
        acceptor_topkl_accuracy, acceptor_auprc = print_topl_statistics(np.asarray(Y_true_1),
                            np.asarray(Y_pred_1), metric_files["topk_acceptor"], type='acceptor', print_top_k=True)
        donor_topkl_accuracy, donor_auprc = print_topl_statistics(np.asarray(Y_true_2),
                            np.asarray(Y_pred_2), metric_files["topk_donor"], type='donor', print_top_k=True)
        for k, v in metric_files.items():
            with open(v, 'a') as f:
                if k == "loss":
                    f.write(f"{running_loss}\n")
                elif k == "topk_acceptor":
                    f.write(f"{acceptor_topkl_accuracy}\n")
                elif k == "topk_donor":
                    f.write(f"{donor_topkl_accuracy}\n")
                elif k == "auprc_acceptor":
                    f.write(f"{acceptor_auprc}\n")
                elif k == "auprc_donor":
                    f.write(f"{donor_auprc}\n")
        wandb.log({
            f'{run_mode}/loss': loss.item(),
            f'{run_mode}/topk_acceptor': acceptor_topkl_accuracy,
            f'{run_mode}/topk_donor': donor_topkl_accuracy,
            f'{run_mode}/auprc_acceptor': acceptor_auprc,
            f'{run_mode}/auprc_donor': donor_auprc,
        })
        print("***************************************\n\n")


def train_epoch(model, h5f, idxs, batch_size, criterion, optimizer, scheduler, device, params, metric_files, run_mode):
    print(f"\033[1m{run_mode.capitalize()}ing model...\033[0m")
    model.train()
    running_loss = 0.0
    np.random.seed(RANDOM_SEED)  # You can choose any number as a seed
    shuffled_idxs = np.random.choice(idxs, size=len(idxs), replace=False)
    print("shuffled_idxs: ", shuffled_idxs)
    batch_ylabel = []
    batch_ypred = []
    print_dict = {}
    for i, shard_idx in enumerate(shuffled_idxs, 1):
        print(f"Shard {i}/{len(shuffled_idxs)}")
        loader = load_data_from_shard(h5f, shard_idx, device, batch_size, params, shuffle=True)
        pbar = tqdm(loader, leave=False, total=len(loader), desc=f'Shard {i}/{len(shuffled_idxs)}')
        for batch in pbar:
            DNAs, labels = batch[0].to(device), batch[1].to(device)
            # print("\n\tDNAs.shape: ", DNAs.shape)
            # print("\tlabels.shape: ", labels.shape)
            DNAs, labels = clip_datapoints(DNAs, labels, params["CL"], 2)
            DNAs, labels = DNAs.to(torch.float32).to(device), labels.to(torch.float32).to(device)
            # print("\n\tAfter clipping DNAs.shape: ", DNAs.shape)
            # print("\tAfter clipping labels.shape: ", labels.shape)
            optimizer.zero_grad()
            yp = model(DNAs)
            loss = categorical_crossentropy_2d(labels, yp)
            # print("loss: ", loss.item())
            # loss = criterion(labels, yp)  # Permuting to [batch_size, sequence_length, num_classes]
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_ylabel.append(labels.detach().cpu())
            batch_ypred.append(yp.detach().cpu())
            print_dict["loss"] = loss.item()
            pbar.set_postfix(print_dict)
            pbar.update(1)
        pbar.close()
    batch_ylabel = torch.cat(batch_ylabel, dim=0)
    batch_ypred = torch.cat(batch_ypred, dim=0)
    is_expr = (batch_ylabel.sum(axis=(1,2)) >= 1).cpu().numpy()
    scheduler.step()
    if np.any(is_expr):
        ############################
        # Topk SpliceAI assessment approach
        ############################
        subset_size = 1000
        indices = np.arange(batch_ylabel[is_expr].shape[0])
        subset_indices = np.random.choice(indices, size=min(subset_size, len(indices)), replace=False)
        Y_true_1 = batch_ylabel[is_expr][subset_indices, 1, :].flatten().cpu().detach().numpy()
        Y_true_2 = batch_ylabel[is_expr][subset_indices, 2, :].flatten().cpu().detach().numpy()
        Y_pred_1 = batch_ypred[is_expr][subset_indices, 1, :].flatten().cpu().detach().numpy()
        Y_pred_2 = batch_ypred[is_expr][subset_indices, 2, :].flatten().cpu().detach().numpy()
        acceptor_topkl_accuracy, acceptor_auprc = print_topl_statistics(np.asarray(Y_true_1),
                            np.asarray(Y_pred_1), metric_files["topk_acceptor"], type='acceptor', print_top_k=True)
        donor_topkl_accuracy, donor_auprc = print_topl_statistics(np.asarray(Y_true_2),
                            np.asarray(Y_pred_2), metric_files["topk_donor"], type='donor', print_top_k=True)
        for k, v in metric_files.items():
            with open(v, 'a') as f:
                if k == "loss":
                    f.write(f"{running_loss}\n")
                elif k == "topk_acceptor":
                    f.write(f"{acceptor_topkl_accuracy}\n")
                elif k == "topk_donor":
                    f.write(f"{donor_topkl_accuracy}\n")
                elif k == "auprc_acceptor":
                    f.write(f"{acceptor_auprc}\n")
                elif k == "auprc_donor":
                    f.write(f"{donor_auprc}\n")
        wandb.log({
            f'{run_mode}/loss': loss.item(),
            f'{run_mode}/topk_acceptor': acceptor_topkl_accuracy,
            f'{run_mode}/topk_donor': donor_topkl_accuracy,
            f'{run_mode}/auprc_acceptor': acceptor_auprc,
            f'{run_mode}/auprc_donor': donor_auprc,
        })
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
    parser.add_argument('--model', '-m', default="SpliceAI", type=str)
    args = parser.parse_args()
    print("args: ", args, file=sys.stderr)
    project_root = args.project_root
    project_name = args.project_name
    sequence_length = 5000
    flanking_size = int(args.flanking_size)
    exp_num = args.exp_num
    training_target = args.training_target
    model_arch = args.model
    assert int(flanking_size) in [80, 400, 2000, 10000]
    # assert training_target in ["RefSeq", "MANE", "SpliceAI", "SpliceAI27"]
    if args.disable_wandb:
        os.environ['WANDB_MODE'] = 'disabled'
    wandb.init(project=f'{project_name}', reinit=True)
    device = setup_device()
    print("device: ", device, file=sys.stderr)
    model_output_base, log_output_train_base, log_output_val_base, log_output_test_base = initialize_paths(project_root, project_name, sequence_length, flanking_size, exp_num, training_target, sequence_length, model_arch)
    print("* model_output_base: ", model_output_base, file=sys.stderr)
    print("* log_output_train_base: ", log_output_train_base, file=sys.stderr)
    print("* log_output_val_base: ", log_output_val_base, file=sys.stderr)
    print("* log_output_test_base: ", log_output_test_base, file=sys.stderr)
    training_dataset = args.train_dataset
    testing_dataset = args.test_dataset
    print("training_dataset: ", training_dataset, file=sys.stderr)
    print("testing_dataset: ", testing_dataset, file=sys.stderr)
    
    train_h5f = h5py.File(training_dataset, 'r')
    test_h5f = h5py.File(testing_dataset, 'r')
    batch_num = len(train_h5f.keys()) // 2
    print("batch_num: ", batch_num, file=sys.stderr)
    np.random.seed(RANDOM_SEED)  # You can choose any number as a seed
    idxs = np.random.permutation(batch_num)
    train_idxs = idxs[:int(0.9 * batch_num)]
    val_idxs = idxs[int(0.9 * batch_num):]
    test_idxs = np.arange(len(test_h5f.keys()) // 2)
    # train_idxs = idxs[:int(0.1*batch_num)]
    # val_idxs = idxs[int(0.2*batch_num):int(0.25*batch_num)]
    # test_idxs = np.arange(len(test_h5f.keys()) // 10)
    print("train_idxs: ", train_idxs, file=sys.stderr)
    print("val_idxs: ", val_idxs, file=sys.stderr)
    print("test_idxs: ", test_idxs, file=sys.stderr)
    model, criterion, optimizer, scheduler, params = initialize_model_and_optim(device, flanking_size, model_arch)
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
    for epoch in range(EPOCH_NUM):
        print("\n--------------------------------------------------------------")
        print(f">> Epoch {epoch + 1}")
        start_time = time.time()
        train_epoch(model, train_h5f, train_idxs, params["BATCH_SIZE"], criterion, optimizer, scheduler, device, params, train_metric_files, run_mode="train")
        valid_epoch(model, train_h5f, val_idxs, params["BATCH_SIZE"], criterion, device, params, valid_metric_files, run_mode="validation")
        valid_epoch(model, test_h5f, test_idxs, params["BATCH_SIZE"], criterion, device, params, test_metric_files, run_mode="test")

        # run_epoch(model, test_h5f, test_idxs, params["BATCH_SIZE"], criterion, None, None, device, params, test_metric_files, run_mode="test")
        torch.save(model.state_dict(), f"{model_output_base}/model_{epoch}.pt")
        print("--- %s seconds ---" % (time.time() - start_time))
        print("--------------------------------------------------------------")
    train_h5f.close()
    test_h5f.close()

if __name__ == "__main__":
    main()
