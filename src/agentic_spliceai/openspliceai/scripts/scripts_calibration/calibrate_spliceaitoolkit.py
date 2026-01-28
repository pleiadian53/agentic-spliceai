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
# from temperature_scaling import *
from temperature_scaling import ModelWithTemperature
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

<<<<<<< HEAD

=======
>>>>>>> main
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


<<<<<<< HEAD
def valid_epoch(model, h5f, idxs, batch_size, device, params, metric_files, run_mode, sample_freq):
    print(f"\033[1m{run_mode.capitalize()}ing model...\033[0m")
    model.eval()
    running_loss = 0.0
    np.random.seed(RANDOM_SEED)  # You can choose any number as a seed
    shuffled_idxs = np.random.choice(idxs, size=len(idxs), replace=False)    
    print("shuffled_idxs: ", shuffled_idxs)
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
            # print("\n\tDNAs.shape: ", DNAs.shape)
            # print("\tlabels.shape: ", labels.shape)
            DNAs, labels = clip_datapoints(DNAs, labels, params["CL"], 2)
            DNAs, labels = DNAs.to(torch.float32).to(device), labels.to(torch.float32).to(device)
            # print("\n\tAfter clipping DNAs.shape: ", DNAs.shape)
            # print("\tAfter clipping labels.shape: ", labels.shape)
            yp = model(DNAs)
            # if criterion == "cross_entropy_loss":
            loss = categorical_crossentropy_2d(labels, yp)
            # elif criterion == "focal_loss":
            #     loss = focal_loss(labels, yp)
            # Logging loss for every update.
            with open(metric_files["loss_every_update"], 'a') as f:
                f.write(f"{loss.item()}\n")
            wandb.log({
                f'{run_mode}/loss_every_update': loss.item(),
            })
            running_loss += loss.item()
            # print("loss: ", loss.item())
            batch_ylabel.append(labels.detach().cpu())
            batch_ypred.append(yp.detach().cpu())
            print_dict["loss"] = loss.item()
            pbar.set_postfix(print_dict)
            pbar.update(1)
            batch_idx += 1
        if i == 5:
            break
        pbar.close()
    model_evaluation(batch_ylabel, batch_ypred, metric_files, run_mode)


=======
>>>>>>> main
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


<<<<<<< HEAD

############################################
# Model calibration
############################################
def fit_temperature_scaling(model, h5f, idxs, device, batch_size, params, train=False):
    model.eval()  # Ensure the model is in evaluation mode
    logits_list = []
    labels_list = []
    model.eval()
    running_loss = 0.0
    np.random.seed(RANDOM_SEED)  # You can choose any number as a seed
    shuffled_idxs = np.random.choice(idxs, size=len(idxs), replace=False)    
    print("shuffled_idxs: ", shuffled_idxs)
    batch_idx = 0
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
            with torch.no_grad():
                logits = model(DNAs)  # Get raw logits from your model
                logits_list.append(logits.detach().cpu())
                labels_list.append(labels.detach().cpu())
        # if i == 3:
        #     break
    # Concatenate all collected logits and true labels
    logits = torch.cat(logits_list).to(device)
    labels = torch.cat(labels_list).to(device)
    print("logits: ", logits)
    print("labels: ", labels)
    if train == False:
        return None, logits, labels

    # Initialize the temperature scaling model with a reasonable starting temperature, e.g., 1.0
    temp_model = TemperatureScaling().to(device)
    temp_model.train()
    
    # Define optimizer for temperature parameter
    optimizer = optim.LBFGS([temp_model.temperature], lr=0.01, max_iter=50)

    # # Define the closure for LBFGS optimizer
    # def nll_closure():
    #     optimizer.zero_grad()
    #     scaled_logits = temp_model(logits)
    #     loss = F.cross_entropy(scaled_logits.view(-1, 3), labels.view(-1, 3))
    #     loss.backward()
    #     return loss

    def nll_closure():
        optimizer.zero_grad()
        scaled_logits = temp_model(logits)
        # Define weights for each class - higher for minority classes
        class_weights = torch.tensor([1., 1000., 1000.]).to(device)  # Adjust weights as appropriate
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fn(scaled_logits.view(-1, 3), labels.view(-1, 3).max(1)[1])  # Assuming labels are one-hot encoded
        loss.backward()
        return loss

    # Run optimizer to fit the temperature parameter
    optimizer.step(nll_closure)
    # Return the fitted temperature scaling model
    return temp_model, logits, labels



=======
>>>>>>> main
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

<<<<<<< HEAD
=======

>>>>>>> main
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
    SAMPLE_FREQ = 1000
    print("\n--------------------------------------------------------------")
    start_time = time.time()
    BATCH_SIZE = 36
    
    
    
    ##########################################
    # Temperature scaling
    ##########################################    
    scaled_model = ModelWithTemperature(model)
    scaled_model.set_temperature(model, test_h5f, test_idxs, device, params["BATCH_SIZE"], params, train=True)
    # scaled_model = ModelWithTemperature(model).to(device)
    # scaled_model.load_state_dict(torch.load("temp_model.pt"))
    # scaled_model = scaled_model.to(device)
    # scaled_model.set_temperature(model, train_h5f, val_idxs, device, params["BATCH_SIZE"], params, train=False)
    # print("logits: ", logits.shape)
    # print("labels: ", labels.shape)
    torch.save(scaled_model.state_dict(), f"temp_model.pt")
    print("scaled_model: ", scaled_model)

    # Apply temperature scaling
    logits = scaled_model.logits
    labels = scaled_model.labels
    logits_scaled = None
    with torch.no_grad():
        logits_scaled = scaled_model.temperature_scale(logits)

    print("logits_scaled: ", logits_scaled)
    print("logits_scaled: ", logits_scaled.shape)
    print("logits: ", logits)
    print("logits: ", logits.shape)
    print("labels: ", labels)
    print("labels: ", labels.shape)

    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(logits, dim=1)
    probs_scaled = torch.nn.functional.softmax(logits_scaled, dim=1)
    print("probs: ", probs)
    print("probs: ", probs.shape)
    print("probs_scaled: ", probs_scaled)
    print("probs_scaled: ", probs_scaled.shape)

    # Convert tensors to CPU before converting to NumPy for use with sklearn's log_loss
    probs = probs.detach().cpu().numpy()
    probs_scaled = probs_scaled.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    # ##########################################
    # # Plotting the score distribution
    # ##########################################
    # score_frequency_distribution(probs, probs_scaled, labels, index=0)
    # score_frequency_distribution(probs, probs_scaled, labels, index=1)
    # score_frequency_distribution(probs, probs_scaled, labels, index=2)
    # Assuming labels are one-hot encoded, we convert them to class indices for log_loss
    # Calculate log-loss
    log_loss_before = log_loss(labels, probs)
    log_loss_after = log_loss(labels, probs_scaled)
    # print("probs_after_reshaped: ", probs_after_reshaped)
    # print("labels_indices: ", labels_indices)
    print("Log-loss of")
    print(f" * uncalibrated classifier: {log_loss_before:.8f}")
    print(f" * calibrated classifier: {log_loss_after:.8f}")


    ##########################################
    # Plotting the calibration curve
    ##########################################
    # Compute the calibration curve for each class 
    classes = ["Non-splice site", "Acceptor site", "Donor site"]
    from sklearn.calibration import calibration_curve 
    calibration_curve_values = [] 
    calibration_curve_scaled_values = [] 
    for i in range(3): 
        curve = calibration_curve(labels == i,  
                                probs[:, i],  
                                n_bins=30,  
                                pos_label=True)
        scaled_curve = calibration_curve(labels == i,  
                                probs_scaled[:, i],  
                                n_bins=30,  
                                pos_label=True)  
        calibration_curve_values.append(curve)
        calibration_curve_scaled_values.append(scaled_curve) 
    
    # Plot the calibration curves 
    fig, axs = plt.subplots(1, 3, figsize=(17,5)) 
    for i in range(3): 
        axs[i].plot(calibration_curve_values[i][1],  
                    calibration_curve_values[i][0],  
                    marker='o', label='Original probability') 
        axs[i].plot(calibration_curve_scaled_values[i][1],  
                    calibration_curve_scaled_values[i][0],  
                    marker='o', label='Calibrated probability') 
        axs[i].plot([0, 1], [0, 1], linestyle='--') 
        axs[i].set_xlim([0, 1]) 
        axs[i].set_ylim([0, 1]) 
        axs[i].set_title(f"{classes[i]}", fontsize = 17) 
        axs[i].set_xlabel("Predicted probability", fontsize = 15) 
        axs[i].set_ylabel("Empirical probability", fontsize = 15) 
        axs[i].legend()
    plt.tight_layout() 
    plt.savefig('calibration_curve.png')




<<<<<<< HEAD
=======
    
    
>>>>>>> main
    ##########################################
    # Plotting the score calibration
    ##########################################
    # Plot a random subset of the data for clarity
    sample_size_per_class = 1000 # for example, 10000 positions
    # Initialize an empty list to collect sampled indices
    balanced_subset_indices = []
    # Iterate over each class and sample indices
    for class_id in range(3):  # Assuming 3 classes
        class_indices = np.where(labels == class_id)[0]
        sampled_indices = np.random.choice(class_indices, size=sample_size_per_class, replace=False)
        balanced_subset_indices.append(sampled_indices)
        print("class_id: ", class_id, "; class_indices: ", len(class_indices))
    # Concatenate sampled indices from each class
    balanced_subset_indices = np.concatenate(balanced_subset_indices)
    # Visualize the changes in predicted probabilities
    plt.figure(figsize=(10, 10))
    colors = ["r", "g", "b"]  # Assuming three classes
    for i in balanced_subset_indices:
        print("color: ", colors[labels[i]])
        plt.arrow(
            probs[i, 1],
            probs[i, 2],
            probs_scaled[i, 1] - probs[i, 1],
            probs_scaled[i, 2] - probs[i, 2],
            color=colors[labels[i]],
            head_width=1e-2,
        )

    # # Plot perfect predictions, at each vertex
    # plt.plot([1.0], [0.0], "ro", ms=20, label="None-splice site")
    # plt.plot([0.0], [1.0], "go", ms=20, label="Donor site")
    # plt.plot([0.0], [0.0], "bo", ms=20, label="Acceptor site")
    plt.plot([0.0], [0.0], "ro", ms=20, label="None-splice site")
    plt.plot([1.0], [0.0], "go", ms=20, label="Donor site")
    plt.plot([0.0], [1.0], "bo", ms=20, label="Acceptor site")

    # Plot boundaries of unit simplex
    plt.plot([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], "k", label="Simplex")
    # Annotate points 6 points around the simplex, and mid point inside simplex
    plt.annotate(
        r"($\frac{1}{3}$, $\frac{1}{3}$, $\frac{1}{3}$)",
        xy=(1.0 / 3, 1.0 / 3),
        xytext=(1.0 / 3, 0.23),
        xycoords="data",
        arrowprops=dict(facecolor="black", shrink=0.05),
        horizontalalignment="center",
        verticalalignment="center",
    )
    plt.plot([1.0 / 3], [1.0 / 3], "ko", ms=5)
    plt.annotate(
        r"($\frac{1}{2}$, $0$, $\frac{1}{2}$)",
        xy=(0.5, 0.0),
        xytext=(0.5, 0.1),
        xycoords="data",
        arrowprops=dict(facecolor="black", shrink=0.05),
        horizontalalignment="center",
        verticalalignment="center",
    )
    plt.annotate(
        r"($0$, $\frac{1}{2}$, $\frac{1}{2}$)",
        xy=(0.0, 0.5),
        xytext=(0.1, 0.5),
        xycoords="data",
        arrowprops=dict(facecolor="black", shrink=0.05),
        horizontalalignment="center",
        verticalalignment="center",
    )
    plt.annotate(
        r"($\frac{1}{2}$, $\frac{1}{2}$, $0$)",
        xy=(0.5, 0.5),
        xytext=(0.6, 0.6),
        xycoords="data",
        arrowprops=dict(facecolor="black", shrink=0.05),
        horizontalalignment="center",
        verticalalignment="center",
    )
    plt.annotate(
        r"($0$, $0$, $1$)",
        xy=(0, 0),
        xytext=(0.1, 0.1),
        xycoords="data",
        arrowprops=dict(facecolor="black", shrink=0.05),
        horizontalalignment="center",
        verticalalignment="center",
    )
    plt.annotate(
        r"($1$, $0$, $0$)",
        xy=(1, 0),
        xytext=(1, 0.1),
        xycoords="data",
        arrowprops=dict(facecolor="black", shrink=0.05),
        horizontalalignment="center",
        verticalalignment="center",
    )
    plt.annotate(
        r"($0$, $1$, $0$)",
        xy=(0, 1),
        xytext=(0.1, 1),
        xycoords="data",
        arrowprops=dict(facecolor="black", shrink=0.05),
        horizontalalignment="center",
        verticalalignment="center",
    )
    # Add grid
    plt.grid(False)
    for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        plt.plot([0, x], [x, 0], "k", alpha=0.2)
        plt.plot([0, 0 + (1 - x) / 2], [x, x + (1 - x) / 2], "k", alpha=0.2)
        plt.plot([x, x + (1 - x) / 2], [0, 0 + (1 - x) / 2], "k", alpha=0.2)
    # Add plotting code for simplex boundaries and annotations as in your original script
    # ...
    plt.title("Change of predicted probabilities after temperature scaling")
    plt.xlabel("Probability Donor")
    plt.ylabel("Probability Acceptor")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="best")
    plt.savefig("temperature_scaling_calibration.png", dpi=300)
    plt.clf()



    ##########################################
    # Plotting calibration map
    ##########################################
    # Insert this after your existing plotting code for temperature scaling
    plt.figure(figsize=(10, 10))
    # Assuming you have probs_scaled which are the probabilities after calibration
    # Generate grid of probability values
    p1d = np.linspace(0, 1, 20)
    p0, p1 = np.meshgrid(p1d, p1d)
    p2 = 1 - p0 - p1
    p = np.c_[p0.ravel(), p1.ravel(), p2.ravel()]
    p = p[p[:, 2] >= 0]
    # Assuming scaled_model can predict or has a method to get calibrated probabilities
    # Here, replace this part with the appropriate method call for your calibrated model
    prediction = None
    print("p: ", p)
    print("p.shape: ", p.shape)
    p_tensor = torch.tensor(p, dtype=torch.float32).to(device)

    p_logits = reverse_softmax(p_tensor)

    with torch.no_grad():
        prediction = scaled_model.temperature_scale(p_logits)
    print("prediction before: ", prediction)
    prediction = torch.nn.functional.softmax(prediction, dim=1)
    print("prediction after: ", prediction)

    prediction = prediction.detach().cpu().numpy()
    # # Plot changes in predicted probabilities induced by the calibrators
    # for i in range(prediction.shape[0]):
    #     plt.arrow(
    #         p[i, 0],
    #         p[i, 1],
    #         prediction[i, 0] - p[i, 0],
    #         prediction[i, 1] - p[i, 1],
    #         head_width=1e-2,
    #         color=colors[np.argmax(p[i])],
    #     )
    colors = ["r", "g", "b"]  # Assuming three classes
    for i in range(prediction.shape[0]):
        # print("color: ", colors[labels[i]])
        plt.arrow(
            p[i, 1],
            p[i, 2],
            prediction[i, 1] - p[i, 1],
            prediction[i, 2] - p[i, 2],
            color=colors[np.argmax(p[i])],
            head_width=1e-2,
        )
    plt.plot([0.0], [0.0], "ro", ms=20, label="None-splice site")
    plt.plot([1.0], [0.0], "go", ms=20, label="Donor site")
    plt.plot([0.0], [1.0], "bo", ms=20, label="Acceptor site")

    # Plot the boundaries of the unit simplex and other elements similar to your existing plots
    plt.plot([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], "k", label="Simplex")
    # Add your simplex boundary and grid plotting code here

    # Add grid
    plt.grid(False)
    for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        plt.plot([0, x], [x, 0], "k", alpha=0.2)
        plt.plot([0, 0 + (1 - x) / 2], [x, x + (1 - x) / 2], "k", alpha=0.2)
        plt.plot([x, x + (1 - x) / 2], [0, 0 + (1 - x) / 2], "k", alpha=0.2)
    # Add plotting code for simplex boundaries and annotations as in your original script
    plt.title("Learned temperature scaling calibration map")
    plt.xlabel("Probability Donor")
    plt.ylabel("Probability Acceptor")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="best")
    plt.savefig("temperature_scaling_calibration_map.png")
    plt.clf()
    print("--- %s seconds ---" % (time.time() - start_time))
    print("--------------------------------------------------------------")
    test_h5f.close()

if __name__ == "__main__":
    predict()