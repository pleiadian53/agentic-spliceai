import argparse
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import platform
# from spliceai import *
from utils import *
from constants import *
import h5py
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from keras.models import load_model
import wandb

RANDOM_SEED = 1

def setup_device():
    """Select computation device based on availability."""
    device_str = "cuda" if torch.cuda.is_available() else "mps" if platform.system() == "Darwin" else "cpu"
    return torch.device(device_str)


def initialize_paths(output_dir, project_name, flanking_size, sequence_length):
    """Initialize project directories and create them if they don't exist."""
    MODEL_VERSION = f"{project_name}_{sequence_length}_{flanking_size}"
    model_train_outdir = f"{output_dir}/{MODEL_VERSION}/"
    model_output_base = f"{model_train_outdir}models/"
    log_output_base = f"{model_train_outdir}LOG/"
    log_output_test_base = f"{log_output_base}TEST/"
    for path in [model_output_base, log_output_test_base]:
        os.makedirs(path, exist_ok=True)
    return model_output_base, log_output_test_base


def calculate_metrics(y_true, y_pred):
    """Calculate metrics including precision, recall, f1-score, and accuracy."""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy


def threshold_predictions(y_probs, threshold=0.5):
    """Threshold probabilities to get binary predictions."""
    return (y_probs > threshold).astype(int)


def classwise_accuracy(true_classes, predicted_classes, num_classes):
    accuracies = []
    for i in range(num_classes):
        correct = np.sum((predicted_classes == i) & (true_classes == i))
        total = np.sum(true_classes == i)
        accuracies.append(correct / total if total > 0 else 0)
    return accuracies


def metrics(batch_ypred, batch_ylabel, metric_files):
    # Assuming batch_ylabel and batch_ypred are numpy arrays
    # Convert softmax probabilities to predicted classes
    predicted_classes = np.argmax(batch_ypred, axis=2)  # Ensure this matches your data shape correctly
    true_classes = np.argmax(batch_ylabel, axis=2)  # Adjust the axis if necessary

    # Flatten arrays if they're 2D (for multi-class, not multi-label)
    true_classes_flat = true_classes.flatten()
    predicted_classes_flat = predicted_classes.flatten()

    # Now, calculate the metrics without iterating over each class
    accuracy = accuracy_score(true_classes_flat, predicted_classes_flat)
    precision, recall, f1, _ = precision_recall_fscore_support(true_classes_flat, predicted_classes_flat, average=None)
    class_accuracies = classwise_accuracy(true_classes_flat, predicted_classes_flat, np.max(true_classes_flat) + 1)
    
    # Print overall accuracy (not class-wise)
    overall_accuracy = np.mean(accuracy)
    print(f"Overall Accuracy: {overall_accuracy}")
    for k, v in metric_files.items():
        with open(v, 'a') as f:
            if k == "accuracy":
                f.write(f"{overall_accuracy}\n")
    
    # Iterate over each class to print/save the metrics
    ss_types = ["Non-splice", "acceptor", "donor"]
    for i, (acc, prec, rec, f1_score) in enumerate(zip(class_accuracies, precision, recall, f1)):
        if i < len(ss_types):  # Ensure no index error
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
                        # Append class-wise accuracy under the general accuracy file
                        f.write(f"{acc}\n")


def valid_epoch(model, h5f, idxs, batch_size, device, params, metric_files, run_mode):
    print(f"\033[1m{run_mode.capitalize()}ing model...\033[0m")
    print("--------------------------------------------------------------")
    print("\n\033[1mValidation set metrics:\033[0m")
    batch_ylabel = []
    batch_ypred = []
    Y_true_1 = [[] for t in range(1)]
    Y_true_2 = [[] for t in range(1)]
    Y_pred_1 = [[] for t in range(1)]
    Y_pred_2 = [[] for t in range(1)]
    np.random.seed(RANDOM_SEED)  # You can choose any number as a seed
    shuffled_idxs = np.random.choice(idxs, size=len(idxs), replace=False)    
    print("shuffled_idxs: ", shuffled_idxs)
    for idx in shuffled_idxs[:30]:
        X = h5f['X' + str(idx)]
        Y = h5f['Y' + str(idx)]
        print("\n\tX.shape: ", X.shape)
        print("\tY.shape: ", Y.shape)
        Xc, Yc = clip_datapoints_spliceai27(X, Y, params['CL'], 2)
        Yp = model.predict(Xc, batch_size=params['BATCH_SIZE'])
        batch_ylabel.extend(Yc[0])
        batch_ypred.extend(Yp)
        print("\n\tYp.shape: ", Yp.shape)
        print("\tYc[0].shape: ", Yc[0].shape)
        # loss = categorical_crossentropy_2d(Yc[0], Yp)
        # print("Loss: ", loss)
        if not isinstance(Yp, list):
            Yp = [Yp]
        for t in range(1):
            is_expr = (Yc[t].sum(axis=(1,2)) >= 1)
            Y_true_1[t].extend(Yc[t][is_expr, :, 1].flatten())
            Y_true_2[t].extend(Yc[t][is_expr, :, 2].flatten())
            Y_pred_1[t].extend(Yp[t][is_expr, :, 1].flatten())
            Y_pred_2[t].extend(Yp[t][is_expr, :, 2].flatten())
        # if idx == 1:
        #     break
    batch_ylabel = np.array(batch_ylabel)
    batch_ypred = np.array(batch_ypred)
    print("\n\tbatch_ylabel.shape: ", batch_ylabel.shape)
    print("\tbatch_ypred.shape: ", batch_ypred.shape)

    plt.figure()
    print("\n\033[1mAcceptor:\033[0m")
    for t in range(1):
        acceptor_topk_accuracy, acceptor_auprc, acceptor_auroc = print_topl_statistics(np.asarray(Y_true_1[t]), np.asarray(Y_pred_1[t]), metric_files, ss_type='acceptor', print_top_k=True)
    print("\n\033[1mDonor:\033[0m")
    for t in range(1):
        donor_topk_accuracy, donor_auprc, donor_auroc = print_topl_statistics(np.asarray(Y_true_2[t]), np.asarray(Y_pred_2[t]), metric_files, ss_type='donor',print_top_k=True)
    plt.savefig(metric_files['prc'])
    plt.clf()

    for k, v in metric_files.items():
        with open(v, 'a') as f:
            # if k == "loss_batch":
            #     f.write(f"{loss.item()}\n")
            if k == "donor_topk":
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
    print("--------------------------------------------------------------")
    # model_evaluation(batch_ylabel, batch_ypred, metric_files, run_mode):


def initialize_model_and_optim(flanking_size):
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
    params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE}
    return params


def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', '-p', type=str)
    parser.add_argument('--disable-wandb', '-d', action='store_true', default=False)
    parser.add_argument('--flanking-size', '-f', type=int, default=80)
    parser.add_argument('--test-dataset', '-test', type=str)
    parser.add_argument('--project-name', '-s', type=str)
    parser.add_argument('--model', '-m', default="SpliceAI", type=str)
    args = parser.parse_args()
    print("args: ", args, file=sys.stderr)
    print("Running SpliceAI-toolkit with 'predict' mode")
    
    output_dir = args.output_dir
    project_name = args.project_name
    sequence_length = 5000
    flanking_size = int(args.flanking_size)
    model_path = args.model
    testing_dataset = args.test_dataset
    assert int(flanking_size) in [80, 400, 2000, 10000]
    if args.disable_wandb:
        os.environ['WANDB_MODE'] = 'disabled'
    wandb.init(project=f'{project_name}', reinit=True)
    device = setup_device()
    print("output_dir: ", output_dir, file=sys.stderr)
    print("flanking_size: ", flanking_size, file=sys.stderr)
    print("model_path: ", model_path, file=sys.stderr)
    print("testing_dataset: ", testing_dataset, file=sys.stderr)
    print("device: ", device, file=sys.stderr)
    model_output_base, log_output_test_base = initialize_paths(output_dir, project_name, flanking_size, sequence_length)
    print("* Project name: ", args.project_name, file=sys.stderr)
    print("* Model_output_base: ", model_output_base, file=sys.stderr)
    print("* Log_output_test_base: ", log_output_test_base, file=sys.stderr)

    test_h5f = h5py.File(testing_dataset, 'r')
    np.random.seed(RANDOM_SEED)  # You can choose any number as a seed
    test_idxs = np.arange(len(test_h5f.keys()) // 2)
    print("test_idxs: ", test_idxs, file=sys.stderr)

    params = initialize_model_and_optim(flanking_size)
    # models = [load_model(resource_filename('spliceai', x)) for x in paths]
    model = load_model(model_path)
    print("Model: ", model)

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

    print("\n--------------------------------------------------------------")
    start_time = time.time()
    BATCH_SIZE = 36
    valid_epoch(model, test_h5f, test_idxs, BATCH_SIZE, device, params, test_metric_files, run_mode="test")
    print("--- %s seconds ---" % (time.time() - start_time))
    print("--------------------------------------------------------------")
    test_h5f.close()

if __name__ == "__main__":
    predict()