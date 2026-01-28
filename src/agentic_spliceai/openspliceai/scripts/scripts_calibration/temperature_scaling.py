# import torch
# from torch import nn
# from torch.nn import functional as F

# class TemperatureScaling(nn.Module):
#     def __init__(self, temperature=1.0):
#         super().__init__()
#         self.temperature = nn.Parameter(torch.ones(1) * temperature)

#     def forward(self, logits):
#         return logits / self.temperature

#     def extra_repr(self):
#         return f'temperature={self.temperature.item()}'


import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from utils import *
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

def categorical_crossentropy_2d(y_true, y_pred):
    # print("y_true: ", y_true.shape)
    # print("y_pred: ", y_pred.shape)
    # print("y_true: ", y_true)
    # print("y_pred: ", y_pred)
    # SEQ_WEIGHT = 10
    return - torch.mean(y_true[:, 0]*torch.log(y_pred[:, 0]+1e-10)
                        + y_true[:, 1]*torch.log(y_pred[:, 1]+1e-10)
                        + y_true[:, 2]*torch.log(y_pred[:, 2]+1e-10))

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Compute 2D focal loss.
    
    Parameters:
    - y_true: tensor of true labels.
    - y_pred: tensor of predicted labels.
    - gamma: focusing parameter.
    - alpha: balancing factor.

    Returns:
    - loss: computed focal loss.
    """
    # Ensuring numerical stability
    gamma = 2
    epsilon = 1e-10
    return - torch.mean(y_true[:, 0]*torch.log(y_pred[:, 0]+epsilon) * torch.pow(torch.sub(1, y_pred[:, 0]), gamma)
                        + y_true[:, 1]*torch.log(y_pred[:, 1]+epsilon) * torch.pow(torch.sub(1, y_pred[:, 1]), gamma)
                        + y_true[:, 2]*torch.log(y_pred[:, 2]+epsilon) * torch.pow(torch.sub(1, y_pred[:, 2]), gamma))


def load_data_from_shard(h5f, shard_idx, device, batch_size, params, shuffle=False):
    X = h5f[f'X{shard_idx}'][:].transpose(0, 2, 1)
    Y = h5f[f'Y{shard_idx}'][0, ...].transpose(0, 2, 1)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    ds = TensorDataset(X, Y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True, pin_memory=True)


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        # self.temperature = nn.Parameter(torch.ones(1) * 0.5)
        self.logits = None
        self.scaled_logits = None
        self.labels = None
        self.nll_b = None
        self.ece_b = None
        self.nll_a = None
        self.ece_a = None

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, model, h5f, idxs, device, batch_size, params, train=False):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        # First: collect all the logits and labels for the validation set
        RANDOM_SEED = 40
        logits_list = []
        labels_list = []
        model.eval()
        running_loss = 0.0
        np.random.seed(RANDOM_SEED)  # You can choose any number as a seed
        shuffled_idxs = np.random.choice(idxs, size=len(idxs), replace=False)    
        print("shuffled_idxs: ", shuffled_idxs)
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
            if i == 20:
                break
        # Concatenate all collected logits and true labels
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)
        print("logits: ", logits)
        print("labels: ", labels)
        print("Before flatten logits.shape: ", logits.shape)
        print("Before flatten labels.shape: ", labels.shape)   

        # Flatten the logits and labels to calculate 'nll_criterion' and 'ece_criterion'
        logits = logits.transpose(1, 2).reshape(-1, 3)  # Flattening logits from [size, 3, 5000] to [size*5000, 3]
        # You need to convert them to class indices if they're not already
        labels_ls = labels.transpose(1, 2).reshape(-1, 3)
        labels = labels.argmax(dim=1)  # If labels are one-hot encoded across the 3 classes
        labels = labels.view(-1)  # Flattening labels from [size, 3, 5000] to [size*5000]

        from collections import Counter

        # Count each distinct element in the list
        element_count = Counter(labels.tolist())
        print(element_count)

        print("After logits: ", logits)
        print("After labels: ", labels)
        print("After flatten logits.shape: ", logits.shape)
        print("After flatten labels.shape: ", labels.shape)   
        if train == False:
            # Storing the logits and labels.
            self.logits = logits
            self.labels = labels
            return self

        self.cuda()
        class_weights=torch.tensor([1,5000, 5000],dtype=torch.float).cuda()
        nll_criterion = nn.CrossEntropyLoss(weight=class_weights,reduction='mean').cuda()
        ece_criterion = _ECELoss().cuda()
        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        # before_temperature_fl = focal_loss(logits, labels_ls).item()
        # before_temperature_cel = categorical_crossentropy_2d(logits, labels_ls).item()
        print('Before temperature - NLL: %.8f, ECE: %.8f' % (before_temperature_nll, before_temperature_ece))

        # Prepare for training
        epochs = 1000  # Number of epochs to train for
        # optimizer = optim.SGD([self.temperature], lr=0.1)
        optimizer = optim.Adam([self.temperature], lr=0.001)  # You can adjust the learning rate as needed
        

        # Early stopping parameters
        best_loss = float('inf')
        epochs_since_improvement = 0
        patience = 5  # Number of epochs to wait for improvement before stopping

        for epoch in range(epochs):
            optimizer.zero_grad()
            # loss = nll_criterion(self.temperature_scale(logits), labels)
            loss = ece_criterion(self.temperature_scale(logits), labels)
            # loss = focal_loss(self.temperature_scale(logits), labels_ls)
            # loss = categorical_crossentropy_2d(self.temperature_scale(logits), labels_ls)
            loss.backward()
            optimizer.step()
            
            # Optionally print the loss every epoch
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

            # Check for improvement
            if loss.item() < best_loss:
                best_loss = loss.item()
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
            
            # Early stopping check
            if epochs_since_improvement >= patience:
                print(f"Stopping early at epoch {epoch + 1}. No improvement in loss for {patience} epochs.")
                break

        
        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        # after_temperature_fl = focal_loss(self.temperature_scale(logits), labels_ls).item()
        # after_temperature_cel = categorical_crossentropy_2d(self.temperature_scale(logits), labels_ls).item()
        print('Optimal temperature: %.5f' % self.temperature.item())
        print('After temperature - NLL: %.8f, ECE: %.8f' % (after_temperature_nll, after_temperature_ece))

        # Storing the logits and labels.
        self.logits = logits
        self.scaled_logits = self.temperature_scale(logits)
        self.labels = labels
        self.nll_b = before_temperature_nll
        self.ece_b = before_temperature_ece
        self.nll_a = after_temperature_nll
        self.ece_a = after_temperature_ece
        return self


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece