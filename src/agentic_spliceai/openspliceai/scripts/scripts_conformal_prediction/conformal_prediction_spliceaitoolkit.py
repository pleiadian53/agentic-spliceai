import argparse
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import platform
from spliceai import *
from utils import *
from constants import *
import h5py
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from keras.models import load_model
import wandb
# # from temperature_scaling import *
# from temperature_scaling import ModelWithTemperature
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

RANDOM_SEED = 42

def setup_device():
    """Select computation device based on availability."""
    device_str = "cuda" if torch.cuda.is_available() else "mps" if platform.system() == "Darwin" else "cpu"
    return torch.device(device_str)


def initialize_paths(output_dir, project_name, flanking_size, sequence_length):
    """Initialize project directories and create them if they don't exist."""
    ####################################
    # Modify the model verson here!!
    ####################################
    MODEL_VERSION = f"{project_name}_{sequence_length}_{flanking_size}"
    ####################################
    # Modify the model verson here!!
    ####################################
    model_train_outdir = f"{output_dir}/{MODEL_VERSION}/"
    model_output_base = f"{model_train_outdir}models/"
    log_output_base = f"{model_train_outdir}LOG/"
    log_output_test_base = f"{log_output_base}VAL/"
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


def load_data_from_shard(h5f, shard_idx, device, batch_size, params, shuffle=False):
    X = h5f[f'X{shard_idx}'][:].transpose(0, 2, 1)
    Y = h5f[f'Y{shard_idx}'][0, ...].transpose(0, 2, 1)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    ds = TensorDataset(X, Y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True, pin_memory=True)


def model_evaluation(batch_ylabel, batch_ypred, metric_files, run_mode):
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
        # if criterion == "cross_entropy_loss":
        loss = categorical_crossentropy_2d(batch_ylabel, batch_ypred)
        # elif criterion == "focal_loss":
        #     loss = focal_loss(batch_ylabel, batch_ypred)
        for k, v in metric_files.items():
            with open(v, 'a') as f:
                if k == "loss_batch":
                    f.write(f"{loss.item()}\n")
                elif k == "topk_acceptor":
                    f.write(f"{acceptor_topkl_accuracy}\n")
                elif k == "topk_donor":
                    f.write(f"{donor_topkl_accuracy}\n")
                elif k == "auprc_acceptor":
                    f.write(f"{acceptor_auprc}\n")
                elif k == "auprc_donor":
                    f.write(f"{donor_auprc}\n")
        wandb.log({
            f'{run_mode}/loss_batch': loss.item(),
            f'{run_mode}/topk_acceptor': acceptor_topkl_accuracy,
            f'{run_mode}/topk_donor': donor_topkl_accuracy,
            f'{run_mode}/auprc_acceptor': acceptor_auprc,
            f'{run_mode}/auprc_donor': donor_auprc,
        })
        print("***************************************\n\n")
    batch_ylabel = []
    batch_ypred = []


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


def calibrate_and_predict(model, temp_model, test_loader):
    model.eval() # Ensure the model is in evaluation mode
    temp_model.eval() # Ensure the temperature model is in evaluation mode
    calibrated_probs = []

    for inputs in test_loader:
        inputs = inputs.to(device)  # Move inputs to the appropriate device
        with torch.no_grad():
            logits = model(inputs)  # Obtain logits from the original model
            calibrated_logits = temp_model(logits)  # Calibrate logits
            probs = F.softmax(calibrated_logits, dim=-1)  # Convert logits to probabilities
            calibrated_probs.append(probs)

    calibrated_probs = torch.cat(calibrated_probs).cpu().numpy()  # Concatenate and move to CPU
    return calibrated_probs
############################################
# End of model calibration
############################################


def reverse_softmax(softmax_output, epsilon=1e-12):
    # Using log softmax for numerical stability
    # Adding epsilon to avoid log(0) which is undefined
    logits = torch.log(softmax_output + epsilon)
    # Subtracting the maximum logit from all logits to ensure numerical stability
    # This step is optional and can be customized based on how you handle the reference logit
    logits -= torch.max(logits, dim=-1, keepdim=True).values
    return logits


def score_frequency_distribution(probs, probs_scaled, labels, index=1):
    # Flatten the tensors for simplicity
    probabilities_flat = probs[:, index]
    probabilities_scaled_flat = probs_scaled[:, index]
    print("probabilities_flat.shape: ", probabilities_flat.shape)
    print("probabilities_scaled_flat.shape: ", probabilities_scaled_flat.shape)
    print("labels.shape: ", labels.shape)
    # Filter out cases where the label is zero (i.e., keep only positive cases for Class 1)
    positive_indices = (labels == index)  # True for positive cases
    print("positive_indices: ", positive_indices)
    print("len(positive_indices): ", len(positive_indices))
    probabilities_positive = probabilities_flat[positive_indices]
    probabilities_scaled_positive = probabilities_scaled_flat[positive_indices]
    labels_positive = labels[positive_indices]
    print("len(probabilities_positive): ", len(probabilities_positive))
    print("probabilities_positive: ", probabilities_positive)
    print("len(probabilities_scaled_positive): ", len(probabilities_scaled_positive))
    print("probabilities_scaled_positive: ", probabilities_scaled_positive)
    print("len(labels_positive): ", len(labels_positive))
    print("labels_positive: ", labels_positive)
    # Since we're now only looking at positive cases, the labels will be all ones,
    # and we're primarily interested in the distribution of predicted probabilities.
    # Plotting
    plt.figure(figsize=(10, 6))
    # Probability distribution for Class 1 from logits (only positive cases)
    plt.hist(probabilities_positive, bins=50, alpha=0.5, label='Predicted Probabilities')
    plt.hist(probabilities_scaled_positive, bins=50, alpha=0.5, label='Predicted Calibrated Probabilities')
    # For positive cases, since all labels are 1, we can simply indicate the average or density with a line or annotation
    # plt.axvline(x=np.mean(labels_positive), color='r', linestyle='dashed', linewidth=2, label='Actual Labels')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    if index == 0:
        plt.title('Score Distribution for non-splice site')
        plt.savefig("prob_dist_neither.png")
    if index == 1:
        plt.title('Score Distribution for acceptor site')
        plt.savefig("prob_dist_acceptor.png")
    elif index == 2:
        plt.title('Score Distribution for donor site')
        plt.savefig("prob_dist_donor.png")
    plt.clf()

def valid_epoch(model, h5f, idxs, batch_size, device, params, metric_files, run_mode, indx=1):
    print(f"\033[1m{run_mode.capitalize()}ing model...\033[0m")
    model.eval()
    running_loss = 0.0
    np.random.seed(RANDOM_SEED)
    shuffled_idxs = np.random.choice(idxs, size=len(idxs), replace=False)
    shuffled_idxs = idxs[:indx]
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
    return batch_ylabel, batch_ypred
    # model_evaluation(batch_ylabel, batch_ypred, metric_files, run_mode)




import seaborn as sns

def count_true_false(conformal_sets):
    # Count True values
    # true_set = conformal_sets[conformal_sets == 0]
    # false_set = conformal_sets[conformal_sets > 0]
    # true_count = len(true_set)
    # false_count = len(false_set)    

    true_count = torch.sum(conformal_sets).item()  # .item() converts a tensor with one element to a Python scalar    
    # Count False values by subtracting true_count from the total number of elements
    false_count = conformal_sets.numel() - true_count  # .numel() returns the total number of elements in the tensor

    print(f"Number of plausible predictions (True): {true_count}")
    print(f"Number of implausible predictions (False): {false_count}")
    return true_count, false_count


# Calculate nonconformity scores function
def calculate_nonconformity_scores(labels, predictions):
    # Calculate the negative log likelihood of the true class probabilities as nonconformity scores
    # print("labels: ", labels)
    # print("predictions: ", predictions)
    print("labels.shape: ", labels.shape)
    print("predictions.shape: ", predictions.shape) 
    true_class_probs = torch.sum(labels * predictions, dim=1)  # Summing over the class dimension
    ################
    # Inverse Probability (Confidence Score)
    ################
    nonconformity_scores = 1 - true_class_probs
    
    # ################
    # # Margin Error
    # ################
    # # Step 1: Mask true class to find the maximum incorrect class probability
    # masked_predictions = predictions * (1 - labels)  # Zero out probabilities for the true class
    # max_incorrect_probs = torch.max(masked_predictions, dim=1).values  # [N, L]
    # # Step 2: Compute the nonconformity scores (margin error)
    # nonconformity_scores = 1 - true_class_probs + max_incorrect_probs  # [N, L]

    ################
    # Logarithmic Loss (Cross-Entropy)
    ################
    # nonconformity_scores = -torch.log(true_class_probs + 1e-10)  # Add epsilon to avoid log(0)

    # print("true_class_probs: ", true_class_probs)
    # print("nonconformity_scores: ", nonconformity_scores)
    print("true_class_probs.shape: ", true_class_probs.shape)
    print("nonconformity_scores.shape: ", nonconformity_scores.shape)

    # # Flatten the tensor to a one-dimensional array for plotting
    # probabilities_flat = true_class_probs.flatten()
    # # Plotting the histogram using seaborn for better aesthetics
    # plt.figure(figsize=(10, 6))  # Set the figure size for better readability
    # sns.histplot(probabilities_flat, bins=50, kde=True, color='blue')  # KDE plots a smooth estimate of the distribution
    # plt.title('Distribution of True Class Probabilities')
    # plt.xlabel('Probability')
    # plt.ylabel('Frequency')
    # plt.grid(True)  # Adding a grid for better visualization of scales
    # plt.savefig('./vis/true_class_probabilities.png')  # Save the plot as an image
    return nonconformity_scores


# Calculate threshold for conformal predictions
def conformal_prediction_threshold(nonconformity_scores, alpha=0.15):
    sorted_scores = torch.sort(nonconformity_scores.flatten()).values
    n = len(sorted_scores)
    q_level = np.ceil((n+1)*(1-alpha))/n
    qhat = np.quantile(nonconformity_scores, q_level, method='higher')

    print("sorted_scores: ", sorted_scores)
    print("sorted_scores.shape: ", sorted_scores.shape)
    print("n: ", n)
    print("q_level: ", q_level)
    print("qhat: ", qhat)


    # quantile_index = int((1 - alpha) * len(sorted_scores))
    # threshold = sorted_scores[max(0, quantile_index - 1)]  # Adjust for 0-based index
    # print("sorted_scores: ", sorted_scores)
    # print("sorted_scores.shape: ", sorted_scores.shape)
    # print("quantile_index: ", quantile_index)
    # print("threshold: ", threshold)

    # plt.figure(figsize=(10, 5))
    # plt.plot(sorted_scores.numpy(), label='Nonconformity Scores')
    # plt.axhline(y=qhat.item(), color='r', linestyle='--', label=f'Threshold at {100*(1-alpha)}% Confidence')
    # plt.title('Nonconformity Scores and Threshold')
    # plt.xlabel('Index')
    # plt.ylabel('Nonconformity Score')
    # plt.legend()
    # plt.savefig('./vis/nonconformity_scores.png')
    return qhat


# Function to make conformal predictions
def make_conformal_predictions(predictions, qhat):

    print("(1-qhat): ", (1-qhat))
    conformal_sets = predictions >= (1-qhat)

    # max_probs, predicted_classes = predictions.max(dim=1)  # Assuming dim=1 is the class dimension
    # nonconformity_scores = 1 - max_probs  # Simplest form of nonconformity: 1 - max predicted probability

    # print("max_probs: ", max_probs)
    # print("max_probs: ", max_probs.shape)
    # print("predicted_classes: ", predicted_classes) 
    # print("predicted_classes: ", predicted_classes.shape) 
    # print("nonconformity_scores: ", nonconformity_scores)
    # print("nonconformity_scores: ", nonconformity_scores.shape)

    # conformal_sets = nonconformity_scores <= threshold  # True if within the conformal prediction threshold
    # # print("new_scores: ", new_scores)
    print("conformal_sets: ", conformal_sets)
    print("conformal_sets: ", conformal_sets.shape)
    # return conformal_sets

    true_count, false_count = count_true_false(conformal_sets)
    print("true_count: ", true_count)
    print("false_count: ", false_count)


    plt.figure(figsize=(12, 6))
    plt.imshow(conformal_sets.numpy(), aspect='auto', interpolation='nearest', cmap='gray')
    plt.colorbar()
    plt.title('Heatmap of Plausible Predictions (True=Plausible)')
    plt.xlabel('Position in Sequence')
    plt.ylabel('Sequence Index')
    plt.savefig('./vis/heatmap_plausible_prediction.png')

    # return predicted_classes[conformal_sets], nonconformity_scores[conformal_sets]

    



def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', '-p', type=str)
    parser.add_argument('--disable-wandb', '-d', action='store_true', default=False)
    parser.add_argument('--flanking-size', '-f', type=int, default=80)
    parser.add_argument('--train-dataset', '-train', type=str)
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
    training_dataset = args.train_dataset
    testing_dataset = args.test_dataset
    assert int(flanking_size) in [80, 400, 2000, 10000]
    if args.disable_wandb:
        os.environ['WANDB_MODE'] = 'disabled'
    wandb.init(project=f'{project_name}', reinit=True)
    device = setup_device()
    print("output_dir: ", output_dir, file=sys.stderr)
    print("flanking_size: ", flanking_size, file=sys.stderr)
    print("model_path: ", model_path, file=sys.stderr)
    print("training_dataset: ", training_dataset, file=sys.stderr)
    print("testing_dataset: ", testing_dataset, file=sys.stderr)
    print("device: ", device, file=sys.stderr)
    model_output_base, log_output_test_base = initialize_paths(output_dir, project_name, flanking_size, sequence_length)
    print("* Project name: ", args.project_name, file=sys.stderr)
    print("* Model_output_base: ", model_output_base, file=sys.stderr)
    print("* Log_output_test_base: ", log_output_test_base, file=sys.stderr)

    train_h5f = h5py.File(training_dataset, 'r')
    test_h5f = h5py.File(testing_dataset, 'r')
    batch_num = len(train_h5f.keys()) // 2
    print("Batch_num: ", batch_num, file=sys.stderr)
    np.random.seed(RANDOM_SEED)  # You can choose any number as a seed
    idxs = np.random.permutation(batch_num)
    train_idxs = idxs[:int(0.9 * batch_num)]
    val_idxs = idxs[int(0.9 * batch_num):]
    test_idxs = np.arange(len(test_h5f.keys()) // 2)
    print("test_idxs: ", test_idxs, file=sys.stderr)
    params = initialize_model_and_optim(flanking_size)
    # Loading trained spliceAI model
    model = SpliceAI(params['L'], params['W'], params['AR'])
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    print("Model: ", model)
    print("\n--------------------------------------------------------------")
    start_time = time.time()
    BATCH_SIZE = 36
    
    
    
    ##########################################
    # Conformal prediction
    ##########################################    
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
    batch_ylabel, batch_ypred_logits = valid_epoch(model, train_h5f, val_idxs, BATCH_SIZE, device, params, test_metric_files, 'VAL', 4)
    batch_ylabel = torch.cat(batch_ylabel, dim=0)
    batch_ypred_logits = torch.cat(batch_ypred_logits, dim=0)
    # Apply softmax to convert logits to probabilities
    batch_ypred_softmax = F.softmax(batch_ypred_logits, dim=1)
    print("batch_ylabel: ", (batch_ylabel))
    print("batch_ypred_logits: ", (batch_ypred_logits))
    print("batch_ypred_softmax: ", (batch_ypred_softmax))
    print("batch_ylabel: ", len(batch_ylabel))
    print("batch_ypred_logits: ", len(batch_ypred_logits))
    print("batch_ypred_softmax: ", len(batch_ypred_softmax))
    print("batch_ylabel.shape: ", batch_ylabel.shape)
    print("batch_ypred_logits.shape: ", batch_ypred_logits.shape)
    print("batch_ypred_softmax.shape: ", batch_ypred_softmax.shape)

    # Calculate nonconformity scores for the calibration set
    nonconformity_scores = calculate_nonconformity_scores(batch_ylabel, batch_ypred_softmax)
    threshold = conformal_prediction_threshold(nonconformity_scores)

    batch_ylabel_test, new_prediction_logits = valid_epoch(model, test_h5f, test_idxs, BATCH_SIZE, device, params, test_metric_files, 'TEST', 4)
    batch_ylabel_test = torch.cat(batch_ylabel_test, dim=0)
    new_prediction_logits = torch.cat(new_prediction_logits, dim=0)

    # Apply softmax to convert logits to probabilities
    prediction_softmax_test = F.softmax(new_prediction_logits, dim=1)
    print("batch_ylabel_test: ", batch_ylabel_test.shape)
    print("prediction_softmax_test: ", prediction_softmax_test.shape)

    # Make conformal predictions for new data
    # predicted_classes, plausible_nonconformity_scores = 
    make_conformal_predictions(prediction_softmax_test, threshold)
    
    # print("Conformal Prediction Sets (True indicates inclusion in the prediction set):")
    # print("predicted_classes: ", predicted_classes) 
    # print("predicted_classes: ", predicted_classes.shape) 
    # print("plausible_nonconformity_scores: ", plausible_nonconformity_scores)
    # print("plausible_nonconformity_scores: ", plausible_nonconformity_scores.shape)

    # true_count, false_count = count_true_false(conformal_sets)
    # print("true_count: ", true_count)
    # print("false_count: ", false_count) 
    
    print("--- %s seconds ---" % (time.time() - start_time))
    print("--------------------------------------------------------------")
    test_h5f.close()

if __name__ == "__main__":
    predict()