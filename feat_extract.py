#! /usr/bin/env python3

import torch

import os

from util.args_loader import get_args
from util.data_loader import get_loader_in, get_loader_out
from util.model_loader import get_model
import numpy as np
import torch.nn.functional as F
import time

# --- Ensure cache folder exists ---
os.makedirs("cache", exist_ok=True)

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

loader_in_dict = get_loader_in(args, config_type="eval", split=('train', 'val'))
trainloaderIn, testloaderIn, num_classes = loader_in_dict.train_loader, loader_in_dict.val_loader, loader_in_dict.num_classes
model = get_model(args, num_classes, load_ckpt=True)

batch_size = args.batch_size

FORCE_RUN = False

dummy_input = torch.zeros((1, 3, 32, 32)).to(device)
score, feature_list = model.feature_list(dummy_input)
featdims = [feat.shape[1] for feat in feature_list]

begin = time.time()

for split, in_loader in [('train', trainloaderIn), ('val', testloaderIn)]:

    feat_log = np.zeros((len(in_loader.dataset), sum(featdims)))
    score_log = np.zeros((len(in_loader.dataset), num_classes))
    label_log = np.zeros(len(in_loader.dataset))

    model.eval()
    for batch_idx, (inputs, targets) in enumerate(in_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        start_ind = batch_idx * batch_size
        end_ind = min((batch_idx + 1) * batch_size, len(in_loader.dataset))

        score, feature_list = model.feature_list(inputs)
        out = torch.cat([F.adaptive_avg_pool2d(layer_feat, 1).squeeze() for layer_feat in feature_list], dim=1)

        feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
        label_log[start_ind:end_ind] = targets.data.cpu().numpy()
        score_log[start_ind:end_ind] = score.data.cpu().numpy()

        if batch_idx % 100 == 0:
            print(f"{batch_idx}/{len(in_loader)}")

    # --- Save features, scores, labels separately to avoid ValueError ---
    np.save(f"cache/{args.in_dataset}_{split}_{args.name}_feat.npy", feat_log.T)
    np.save(f"cache/{args.in_dataset}_{split}_{args.name}_score.npy", score_log.T)
    np.save(f"cache/{args.in_dataset}_{split}_{args.name}_label.npy", label_log)
    print(f"{split} features, scores, labels saved successfully in cache folder!")

# --- OOD feature extraction ---
for ood_dataset in args.out_datasets:
    loader_test_dict = get_loader_out(args, dataset=(None, ood_dataset), split=('val'))
    out_loader = loader_test_dict.val_ood_loader

    ood_feat_log = np.zeros((len(out_loader.dataset), sum(featdims)))
    ood_score_log = np.zeros((len(out_loader.dataset), num_classes))

    model.eval()
    for batch_idx, (inputs, _) in enumerate(out_loader):
        inputs = inputs.to(device)
        start_ind = batch_idx * batch_size
        end_ind = min((batch_idx + 1) * batch_size, len(out_loader.dataset))

        score, feature_list = model.feature_list(inputs)
        out = torch.cat([F.adaptive_avg_pool2d(layer_feat, 1).squeeze() for layer_feat in feature_list], dim=1)

        ood_feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
        ood_score_log[start_ind:end_ind, :] = score.data.cpu().numpy()

        if batch_idx % 100 == 0:
            print(f"{batch_idx}/{len(out_loader)}")

    np.save(f"cache/{ood_dataset}_vs_{args.in_dataset}_{args.name}_feat.npy", ood_feat_log.T)
    np.save(f"cache/{ood_dataset}_vs_{args.in_dataset}_{args.name}_score.npy", ood_score_log.T)
    print(f"{ood_dataset} OOD features and scores saved successfully!")

print(f"Feature extraction completed in {time.time() - begin:.2f} seconds")