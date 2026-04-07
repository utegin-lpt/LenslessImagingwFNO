import os
from glob import glob
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
import matplotlib.pyplot as plt

from FNO import FNO2d, DiffuserImageDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the dataset and trained checkpoint

INPUT_FOLDER = '/content/data/dataset/diffused'
TARGET_FOLDER = '/content/data/dataset/ground'
CHECKPOINT = 'FNO_best.pth'

# Resolutions to evaluate. The model is trained only at 128x128 in our case;
# 256 and 512 are unseen during training.
EVAL_RESOLUTIONS = [128, 256, 512]

NUM_PLOT_SAMPLES = 4

# Build the test split (same seed as FNO.py)
# Use the highest evaluation resolution as the "native" loading size,
# so that lower resolutions are obtained by downsampling the same images.

NATIVE_RES = max(EVAL_RESOLUTIONS)

native_transform = transforms.Compose([
    transforms.Resize((NATIVE_RES, NATIVE_RES)),
    transforms.ToTensor(),
])

full_dataset = DiffuserImageDataset(INPUT_FOLDER, TARGET_FOLDER, transform=native_transform)

train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

_, _, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
print(f"Test set size: {len(test_dataset)}")

# Load the trained FNO. Be carefull that modes and width should be same with training.

model = FNO2d(
    modes1=24,
    modes2=24,
    width=34,
    n_layers=6,
    in_channels=3,
    out_channels=3,
).to(device)

state = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(state)
model.eval()

param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Loaded checkpoint '{CHECKPOINT}' ({param_count:,} parameters)")

# Multi-resolution evaluation

def downsample(x, size):
    """Bilinear downsample to (size, size). Matches the paper's protocol."""
    if x.shape[-1] == size and x.shape[-2] == size:
        return x
    return F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)


results = {res: {'psnr': [], 'ssim': []} for res in EVAL_RESOLUTIONS}
plot_samples = {res: [] for res in EVAL_RESOLUTIONS}

# Fresh metric objects per resolution to avoid state leakage
psnr_metrics = {res: PeakSignalNoiseRatio(data_range=1.0).to(device) for res in EVAL_RESOLUTIONS}
ssim_metrics = {res: StructuralSimilarityIndexMeasure(data_range=1.0).to(device) for res in EVAL_RESOLUTIONS}

with torch.no_grad():
    for batch_idx, (diffused_native, gt_native) in enumerate(test_loader):
        diffused_native = diffused_native.to(device)
        gt_native = gt_native.to(device)

        for res in EVAL_RESOLUTIONS:
            diffused = downsample(diffused_native, res)
            gt = downsample(gt_native, res)

            pred = torch.clamp(model(diffused), 0.0, 1.0)

            psnr_val = psnr_metrics[res](pred, gt).item()
            ssim_val = ssim_metrics[res](pred, gt).item()

            results[res]['psnr'].append(psnr_val)
            results[res]['ssim'].append(ssim_val)

            if batch_idx == 0 and len(plot_samples[res]) < NUM_PLOT_SAMPLES:
                n_take = min(NUM_PLOT_SAMPLES - len(plot_samples[res]), diffused.shape[0])
                for k in range(n_take):
                    plot_samples[res].append((
                        diffused[k].cpu().permute(1, 2, 0).numpy(),
                        gt[k].cpu().permute(1, 2, 0).numpy(),
                        pred[k].cpu().permute(1, 2, 0).numpy(),
                    ))

# Multi resolution comparision

print("\n" + "=" * 56)
print("Resolution-Agnostic Inference Performance of FNO")
print("(model trained exclusively at 128 x 128)")
print("=" * 56)
print(f"{'Evaluated at':>16} | {'PSNR (dB)':>12} | {'PSNR std':>10} | {'SSIM':>8}")
print("-" * 56)
for res in EVAL_RESOLUTIONS:
    p = np.array(results[res]['psnr'])
    s = np.array(results[res]['ssim'])
    print(f"{res:>6} x {res:<6} | {p.mean():>12.2f} | {p.std():>10.2f} | {s.mean():>8.4f}")
print("=" * 56)


n_rows = len(EVAL_RESOLUTIONS)
n_cols = NUM_PLOT_SAMPLES * 3  

fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
if n_rows == 1:
    axes = axes[None, :]

col_titles = []
for _ in range(NUM_PLOT_SAMPLES):
    col_titles += ['Diffuser', 'FNO', 'Ground Truth']

for r, res in enumerate(EVAL_RESOLUTIONS):
    samples = plot_samples[res]
    for s_idx, (inp, gt, pred) in enumerate(samples):
        for c_off, img in enumerate([inp, pred, gt]):
            ax = axes[r, s_idx * 3 + c_off]
            ax.imshow(np.clip(img, 0, 1))
            ax.axis('off')
            if r == 0:
                ax.set_title(col_titles[s_idx * 3 + c_off], fontsize=11)
        axes[r, 0].set_ylabel(f"{res}x{res}", fontsize=12, rotation=90, labelpad=10)

plt.tight_layout()
plt.savefig('multires_reconstructions.png', dpi=150, bbox_inches='tight')
print("\nSaved figure to multires_reconstructions.png")
