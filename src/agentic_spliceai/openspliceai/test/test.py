import torch.optim as optim
from openspliceai.train_base.openspliceai import *
from openspliceai.train_base.utils import *
from keras.models import load_model

def initialize_model_and_optim(device, flanking_size, pretrained_model, random_seed, test_target):
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
    print("\033[1mContext nucleotides: %d\033[0m" % (CL))
    print("\033[1mSequence length (output): %d\033[0m" % (SL))
    if test_target == "SpliceAI-Keras":
        model = load_model(pretrained_model)
        params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE, 'RANDOM_SEED': random_seed}
        return model, params
    elif test_target == "OpenSpliceAI":
        # Initialize the model
        model = SpliceAI(L, W, AR).to(device)
        # Print the shapes of the parameters in the initialized model
        print("\nInitialized model parameter shapes:")
        for name, param in model.named_parameters():
            print(f"{name}: {param.shape}", end=", ")

        # Load the pretrained model
        state_dict = torch.load(pretrained_model, map_location=device)

        # Filter out unnecessary keys and load matching keys into model
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

        # Load state dict into the model
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        # Print missing and unexpected keys
        print("\nMissing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

        # Set up optimizer and scheduler
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        print(model, file=sys.stderr)    
        params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE, 'N_GPUS': N_GPUS, 'RANDOM_SEED': random_seed}
        return model, optimizer, scheduler, params


def test(args):
    print("Running OpenSpliceAI with 'test' mode")
    if args.test_target == "OpenSpliceAI":
        device = setup_environment(args)
        log_output_test_base = initialize_test_paths(args)
        test_h5f = load_test_datasets(args)    
        test_idxs = generate_test_indices(args.random_seed, test_h5f)
        model, optimizer, scheduler, params = initialize_model_and_optim(device, args.flanking_size, args.pretrained_model, args.random_seed, args.test_target)
        test_metric_files = create_metric_files(log_output_test_base)
        test_model(model, optimizer, test_h5f, test_idxs, args, device, params, test_metric_files)
        test_h5f.close()
    elif args.test_target == "SpliceAI-Keras":
        device = setup_environment(args)
        log_output_test_base = initialize_test_paths(args)
        test_h5f = load_test_datasets(args)    
        test_idxs = generate_test_indices(args.random_seed, test_h5f)
        
        model, params = initialize_model_and_optim(device, args.flanking_size, args.pretrained_model, args.random_seed, args.test_target)
        test_metric_files = create_metric_files(log_output_test_base)
        print("test_metric_files: ", test_metric_files)
        test_SpliceAI_Keras_model(model, test_h5f, test_idxs, args, params, test_metric_files)
        # valid_epoch_keras(model, test_h5f, test_idxs, args.loss, params, test_metric_files, args.flanking_size, "Test")



