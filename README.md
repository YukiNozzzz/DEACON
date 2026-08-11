# DEACON

Official PyTorch implementation of **Dual-Branch Mutual Teaching for Long-Tailed Partial-Label Learning** (IJCAI 2026).

DEACON addresses long-tailed partial-label learning (LT-PLL), where the class distribution is imbalanced and every training example is associated with a candidate label set that contains the ground-truth label. The method jointly trains a head-dominant branch and a tail-focused branch, lets the two branches exchange supervision, and learns an attention-based ensemble for the final prediction.

The supplementary material is available in [Appendix.pdf](./Appendix.pdf).

## Method overview

The implementation combines the following components:

- **Dual-branch learning:** a head-dominant branch models the empirical class distribution, while a tail-focused branch uses a balanced prior.
- **Mutual disambiguation:** predictions from the head branch are logit-adjusted and used to guide reliable-sample selection for the tail branch.
- **Distribution-aware contrastive learning:** `DECoLoss` models class features with von Mises-Fisher distributions and uses a numerically stable Miller recurrence for the normalization term.
- **Multi-view representation learning:** two weak views and one RandAugment-based strong view are used for supervised contrastive learning and SimSiam-style consistency.
- **Reliable-sample MixUp:** selected samples and their soft labels are mixed using a Beta(4, 4) distribution.
- **Adaptive ensemble:** a learnable attention module combines the two branches at inference time.

## Repository structure

```text
DEACON/
|-- train.py              # Training, evaluation, sample selection, and checkpointing
|-- resnet.py             # ResNet-18 encoders, projection heads, and ensemble module
|-- deco.py               # DECo distribution-aware contrastive loss
|-- utils/
|   |-- utils_data.py     # CIFAR loading and LT-PLL data generation
|   |-- utils_loss.py     # Partial-label and supervised contrastive losses
|   |-- utils_algo.py     # Metrics, schedules, and training utilities
|   |-- cifar10.py        # CIFAR-10 multi-view dataset wrapper
|   |-- cifar100.py       # CIFAR-100 multi-view dataset wrapper
|   |-- randaugment.py    # Strong augmentation operations
|   |-- sun397.py         # Experimental SUN397 loader
|   |-- voc.py            # Experimental PASCAL VOC loader
|   `-- cub200.py         # Experimental CUB-200 loader
`-- Appendix.pdf          # Supplementary material
```

## Requirements

The current implementation requires an NVIDIA GPU and a CUDA-enabled PyTorch installation because tensors and models are moved to CUDA directly.

Create an environment and install PyTorch using the command appropriate for your CUDA version from the [official PyTorch installation guide](https://pytorch.org/get-started/locally/). Then install the remaining dependencies:

```bash
conda create -n deacon python=3.10 -y
conda activate deacon

# Install torch and torchvision for your CUDA version first.
pip install numpy scipy scikit-learn matplotlib seaborn pandas pillow
```

No exact package lock file is included in this release. Record the working environment before a reproduction run, for example with `pip freeze > environment.txt`.

## Data preparation

### CIFAR-10 and CIFAR-100

CIFAR-10/100 are downloaded automatically by `torchvision` into `--data_dir`. On the first run, the code also generates a long-tailed training split and partial candidate labels, then caches them in:

```text
<data_dir>/pre-processed-data/
```

The cache filename records the dataset, partial-label rate, imbalance type, imbalance ratio, and random seed. Delete the corresponding cache file if you intentionally need to regenerate that configuration.

For the default uniform partial-label setting, every incorrect label is independently added to the candidate set with probability `--partial_rate`, while the ground-truth label is always retained. Passing a negative partial rate activates the label-dependent candidate-label generator in the code.

An example directory after the first run is:

```text
data/
|-- cifar-10-batches-py/
`-- pre-processed-data/
    `-- cifar10_0.5_imb_exp100.0_sd1.npy
```

## Training

Run commands from the repository root.

### CIFAR-10-LT

```bash
python -u train.py \
  --dataset cifar10 \
  --partial_rate 0.5 \
  --imb_ratio 100 \
  --exp-dir experiment/CIFAR10 \
  --data_dir ./data \
  --epochs 800 \
  --batch-size 256 \
  --lr 0.01 \
  --wd 1e-3 \
  --t 2 \
  --save_ckpt
```

### CIFAR-100-LT

```bash
python -u train.py \
  --dataset cifar100 \
  --partial_rate 0.05 \
  --imb_ratio 20 \
  --exp-dir experiment/CIFAR100 \
  --data_dir ./data \
  --epochs 800 \
  --batch-size 256 \
  --lr 0.01 \
  --wd 1e-3 \
  --t 2 \
  --save_ckpt
```

### PASCAL VOC

```bash
python -u train.py \
  --dataset voc \
  --partial_rate 0 \
  --imb_ratio 1 \
  --exp-dir experiment/VOC \
  --data_dir ./data \
  --epochs 200 \
  --batch-size 128 \
  --lr 0.01 \
  --wd 1e-3 \
  --t 0.99 \
  --save_ckpt
```

Arguments not shown in these commands use the defaults defined in `train.py`.

## Important arguments

| Argument | Default | Description |
| --- | ---: | --- |
| `--dataset` | `cifar10` | Dataset name. Reproduction configurations are provided for `cifar10`, `cifar100`, and `voc`. |
| `--data_dir` | `../codes/data/` | Dataset root and location of the generated preprocessing cache. |
| `--exp-dir` | `experiment/cifar10` | Root directory for logs and checkpoints. |
| `--epochs` | `800` | Total number of training epochs. |
| `--batch-size` | `256` | Training batch size. |
| `--lr` | `0.01` | Initial SGD learning rate. |
| `-lr_decay_epochs` | `700,800` | Comma-separated learning-rate decay milestones. |
| `-lr_decay_rate` | `0.1` | Multiplicative learning-rate decay factor. |
| `--partial_rate` | `0.3` | Probability of adding each incorrect label in the uniform partial-label setting. |
| `--hierarchical` | `False` | Use hierarchical candidate labels for CIFAR-100. |
| `--imb_type` | `exp` | Long-tail profile: `exp` or `step`. |
| `--imb_ratio` | `100` | Maximum-to-minimum class imbalance ratio. |
| `--alpha_range` | `0.2,0.6` | Start and end ratios used by reliable-sample selection. |
| `--eta` | `0.9` | Final weight of the reliable-sample loss. |
| `--t` | `2` | Logit-adjustment strength. |
| `--e` | `50` | Ramp-up length for `alpha` and `eta`. |
| `--prot_start` | `80` | Epoch at which cross-branch reliable-sample filtering starts. |
| `--warmup_epoch_head` | `80` | Epoch at which the head contrastive term is enabled. |
| `--warmup_epoch_tail` | `100` | Epoch at which the tail DECo term is enabled. |
| `--temp` | `0.1` | Temperature used by DECo. |
| `--feat_dim` | `128` | Feature dimension expected by DECo. |
| `--seed` | `1` | Random seed for Python, NumPy, and PyTorch. |
| `--save_ckpt` | off | Save the latest and best ensemble checkpoints. |

## Outputs

Each run creates a configuration-specific directory below `--exp-dir`, for example:

```text
experiment/CIFAR10/
`-- cifar10_p0.5_alpha0.2,0.6_tau2.0_ep800_e50_imb_exp100.0_sd_1/
    |-- result.log
    |-- checkpoint.pth.tar
    `-- checkpoint_best_ens.pth.tar
```

`result.log` contains overall top-1 accuracy and many-/medium-/few-shot accuracy for the tail branch, head branch, and learned ensemble. Checkpoints are produced only when `--save_ckpt` is enabled.

## Current scope and known limitations

- The current training path is GPU-only; CPU execution is not implemented.
- Reproduction configurations are provided for CIFAR-10, CIFAR-100, and PASCAL VOC.
- The PASCAL VOC loader expects the CSV and image layout encoded in `utils/voc.py`.
- SUN397 and CUB-200 data utilities are included for research use, but no official training command is provided here.
- `--resume` is present in the command-line interface, but the loading keys do not currently match the multi-model checkpoint keys written by `--save_ckpt`; resuming a saved dual-branch run therefore requires a small code adjustment.
- Pretrained checkpoints and a pinned environment file are not included.

## Citation

If you find this repository useful, please cite the paper. The BibTeX entry will be added after the publication metadata is available.

## Acknowledgements

This implementation uses PyTorch/torchvision data utilities and includes supervised contrastive learning, SimSiam-style consistency, RandAugment, MixUp, and von Mises-Fisher feature modeling components.
