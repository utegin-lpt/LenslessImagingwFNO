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
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import matplotlib.pyplot as plt

# Which trained model to evaluate: 'fno' or 'unet'. 

MODEL = 'fno'

if MODEL == 'fno':
    from FNO import FNO2d, DiffuserImageDataset
else:
    from unet import UNet, DiffuserImageDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the dataset and trained checkpoint

INPUT_FOLDER = '/content/data/dataset/diffused'
TARGET_FOLDER = '/content/data/dataset/ground'
CHECKPOINT = 'FNO_best.pth' if MODEL == 'fno' else 'UNet_best.pth'

MODEL_LABEL = 'FNO' if MODEL == 'fno' else 'U-Net'

# Resolutions to evaluate, as (height, width). The model is trained only at 90x160 in
# our case; 180x320 and 270x480 are unseen during training. Every step is an exact
# multiple of 90x160, so the aspect ratio never changes, and 270x480 is the native
# resolution of the captured pairs.
EVAL_RESOLUTIONS = [(90, 160), (180, 320), (270, 480)]

NUM_PLOT_SAMPLES = 4

# Build the test split (same seed as FNO.py)
# Use the highest evaluation resolution as the "native" loading size,
# so that lower resolutions are obtained by downsampling the same images.

NATIVE_RES = max(EVAL_RESOLUTIONS)

native_transform = transforms.Compose([
    transforms.Resize(NATIVE_RES),
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

# Load the trained model. Be carefull that the arguments should be same with training.

if MODEL == 'fno':
    model = FNO2d(
        modes=24,
        width=29,
        n_layers=4,
        corners=2,
        init='sqrt_cin',
        norm='group',
        input_norm='max',
        use_grid=True,
        padding=0.0,
        mlp_ratio=1.0,
        residual=False,
        proj_hidden=128,
        in_ch=3,
        out_ch=3,
    ).to(device)
else:
    model = UNet(
        depth=3,
        base=64,
        k=3,
        ch_max=512,
        center_convs=2,
        dilation=4,
        center_dilations=(4, 4),
        pool='max',
        padding_mode='zeros',
        bottleneck='none',
        dropout=0.5,
        input_maxnorm=False,
        in_ch=3,
        out_ch=3,
    ).to(device)

state = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(state)
model.eval()

param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Loaded checkpoint '{CHECKPOINT}' ({param_count:,} parameters)")

# Multi-resolution evaluation

def downsample(x, size):
    """Bilinear downsample to (height, width). Matches the paper's protocol."""
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    return F.interpolate(x, size=tuple(size), mode='bilinear', align_corners=False)


results = {res: {'psnr': [], 'ssim': [], 'lpips': [], 'mse': []} for res in EVAL_RESOLUTIONS}
plot_samples = {res: [] for res in EVAL_RESOLUTIONS}

# Metric objects per resolution to avoid state leakage
psnr_metrics = {res: PeakSignalNoiseRatio(data_range=1.0).to(device) for res in EVAL_RESOLUTIONS}
ssim_metrics = {res: StructuralSimilarityIndexMeasure(data_range=1.0).to(device) for res in EVAL_RESOLUTIONS}
lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)

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
            lpips_val = lpips_metric(pred, gt).item()
            lpips_metric.reset()
            mse_val = F.mse_loss(pred, gt).item()

            results[res]['psnr'].append(psnr_val)
            results[res]['ssim'].append(ssim_val)
            results[res]['lpips'].append(lpips_val)
            results[res]['mse'].append(mse_val)

            if batch_idx == 0 and len(plot_samples[res]) < NUM_PLOT_SAMPLES:
                n_take = min(NUM_PLOT_SAMPLES - len(plot_samples[res]), diffused.shape[0])
                for k in range(n_take):
                    plot_samples[res].append((
                        diffused[k].cpu().permute(1, 2, 0).numpy(),
                        gt[k].cpu().permute(1, 2, 0).numpy(),
                        pred[k].cpu().permute(1, 2, 0).numpy(),
                    ))

# Multi resolution comparision

print("\n" + "=" * 78)
print(f"Resolution-Agnostic Inference Performance of {MODEL_LABEL}")
print("(model trained exclusively at 90 x 160)")
print("=" * 78)
print(f"{'Evaluated at':>16} | {'PSNR (dB)':>12} | {'PSNR std':>10} | {'SSIM':>8} | {'LPIPS':>8} | {'MSE':>9}")
print("-" * 78)
for res in EVAL_RESOLUTIONS:
    p = np.array(results[res]['psnr'])
    s = np.array(results[res]['ssim'])
    l = np.array(results[res]['lpips'])
    m = np.array(results[res]['mse'])
    print(f"{res[0]:>6} x {res[1]:<6} | {p.mean():>12.2f} | {p.std():>10.2f} | {s.mean():>8.4f}"
          f" | {l.mean():>8.4f} | {m.mean():>9.5f}")
print("=" * 78)
print("PSNR / SSIM: higher is better.  LPIPS / MSE: lower is better.")


n_rows = len(EVAL_RESOLUTIONS)
n_cols = NUM_PLOT_SAMPLES * 3

fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2 * n_rows))
if n_rows == 1:
    axes = axes[None, :]

col_titles = []
for _ in range(NUM_PLOT_SAMPLES):
    col_titles += ['Diffuser', MODEL_LABEL, 'Ground Truth']

for r, res in enumerate(EVAL_RESOLUTIONS):
    samples = plot_samples[res]
    for s_idx, (inp, gt, pred) in enumerate(samples):
        for c_off, img in enumerate([inp, pred, gt]):
            ax = axes[r, s_idx * 3 + c_off]
            ax.imshow(np.clip(img, 0, 1))
            ax.axis('off')
            if r == 0:
                ax.set_title(col_titles[s_idx * 3 + c_off], fontsize=11)
        axes[r, 0].set_ylabel(f"{res[0]}x{res[1]}", fontsize=12, rotation=90, labelpad=10)

plt.tight_layout()
plt.savefig('multires_reconstructions.png', dpi=150, bbox_inches='tight')
print("\nSaved figure to multires_reconstructions.png")
