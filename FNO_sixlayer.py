import os
import math
from glob import glob
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the dataset
# dataset = ...

class DiffuserImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform

        self.input_images = sorted(glob(os.path.join(input_dir, '*.*')))
        self.target_images = sorted(glob(os.path.join(target_dir, '*.*')))


    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        in_img = Image.open(self.input_images[idx]).convert('RGB')
        tgt_img = Image.open(self.target_images[idx]).convert('RGB')

        if self.transform:
            in_img = self.transform(in_img)
            tgt_img = self.transform(tgt_img)

        return in_img, tgt_img

# 90x160 is exactly 270x480 / 3, so the aspect ratio of the sensor is preserved.
# The measurement and the ground truth get the same geometric treatment,
# identical resize factors leave y = h * x a convolution with a
# rescaled PSF, different factors do not.

transform = transforms.Compose([
    transforms.Resize((90, 160)),
    transforms.ToTensor()
])

INPUT_FOLDER = '/content/data/dataset/diffused'
TARGET_FOLDER = '/content/data/dataset/ground'

full_dataset = DiffuserImageDataset(INPUT_FOLDER, TARGET_FOLDER, transform=transform)

train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

# Define the Fourier Neural Operator (FNO) model.
# Every knob preserves discretization invariance:
#   * spectral weights live on fixed low-frequency modes (grid-independent)
#   * all spatial convs are 1x1 by default (pointwise)
#   * GroupNorm / InstanceNorm normalize per sample over space
#   * the coordinate grid uses normalized [0,1] coordinates
#   * domain padding is a fixed *fraction* of the input size

def _pair(m):
    return (m, m) if isinstance(m, int) else tuple(m)


class SpectralConv2d(nn.Module):
    """Learned complex weights on the lowest retained frequencies.

    modes: int or (m1, m2). Anisotropic budgets matter at 90x160, where the
        vertical rfft axis holds 45 nonnegative frequencies but the horizontal 81.
    corners=2 fills out_ft[:, :, :m1, :m2] AND out_ft[:, :, -m1:, :m2]
        (Li et al.'s FNO2d); corners=1 reproduces the single-corner variant.
    init: 'sqrt_cin' | 'prev' | 'fno_ref'
    """

    def __init__(self, cin, cout, modes, corners=2, init='sqrt_cin'):
        super().__init__()
        m1, m2 = _pair(modes)
        self.cin, self.cout, self.m1, self.m2, self.corners = cin, cout, m1, m2, corners
        if init == 'sqrt_cin':
            s, uniform = 1.0 / math.sqrt(cin), False
        elif init == 'fno_ref':
            s, uniform = 1.0 / (cin * cout), True
        else:  # 'prev'
            s, uniform = 1.0 / math.sqrt(cin * cout), False

        def w():
            t = torch.rand(cin, cout, m1, m2, 2) if uniform else torch.randn(cin, cout, m1, m2, 2)
            return nn.Parameter(s * t)

        self.w1 = w()
        self.w2 = w() if corners == 2 else None

    def forward(self, x):
        B, C, H, W = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = torch.fft.rfft2(x.float(), norm='ortho')
            m1 = min(self.m1, max(H // 2, 1))
            m2 = min(self.m2, W // 2 + 1)
            out = torch.zeros(B, self.cout, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
            w1 = torch.view_as_complex(self.w1)[:, :, :m1, :m2]
            out[:, :, :m1, :m2] = torch.einsum('bixy,ioxy->boxy', xf[:, :, :m1, :m2], w1)
            if self.w2 is not None:
                assert 2 * m1 <= H, f"corners overlap: 2*{m1} > {H}; lower modes or add padding"
                w2 = torch.view_as_complex(self.w2)[:, :, :m1, :m2]
                out[:, :, -m1:, :m2] = torch.einsum('bixy,ioxy->boxy', xf[:, :, -m1:, :m2], w2)
            return torch.fft.irfft2(out, s=(H, W), norm='ortho')


def make_norm(kind, ch):
    if kind in (None, 'none'):
        return nn.Identity()
    if kind == 'group':
        g = min(8, ch)
        while ch % g:            
            g -= 1
        return nn.GroupNorm(g, ch)
    if kind == 'instance':
        return nn.InstanceNorm2d(ch, affine=True)
    raise ValueError(kind)


class FNOBlock(nn.Module):
    """v -> act( norm( [mlp](R.F[v]) + W v ) ), optionally + v (outer residual)."""

    def __init__(self, width, modes, corners, init, norm,
                 bypass_kernel=1, mlp_ratio=0.0, residual=False):
        super().__init__()
        self.spec = SpectralConv2d(width, width, modes, corners, init)
        self.byp = nn.Conv2d(width, width, bypass_kernel, padding=bypass_kernel // 2)
        self.mlp = None
        if mlp_ratio and mlp_ratio > 0:
            h = max(int(round(width * mlp_ratio)), 1)
            self.mlp = nn.Sequential(nn.Conv2d(width, h, 1), nn.GELU(), nn.Conv2d(h, width, 1))
        self.n = make_norm(norm, width)
        self.residual = residual

    def forward(self, x):
        s = self.spec(x)
        if self.mlp is not None:
            s = self.mlp(s)
        y = F.gelu(self.n(s + self.byp(x)))
        return x + y if self.residual else y


class FNO2d(nn.Module):
    def __init__(self, modes=32, width=64, n_layers=6, corners=2, init='sqrt_cin',
                 norm='group', input_norm='max', use_grid=True, padding=0.0,
                 mlp_ratio=1.0, residual=False, bypass_kernel=1,
                 in_ch=3, out_ch=3, lift_hidden=0, proj_hidden=128,
                 out_sigmoid=True):
        super().__init__()
        self.input_norm, self.use_grid, self.padding = input_norm, use_grid, float(padding)
        self.out_sigmoid = out_sigmoid
        cin = in_ch + (2 if use_grid else 0)
        if lift_hidden and lift_hidden > 0:
            self.lift = nn.Sequential(nn.Conv2d(cin, lift_hidden, 1), nn.GELU(),
                                      nn.Conv2d(lift_hidden, width, 1))
        else:
            self.lift = nn.Conv2d(cin, width, 1)
        self.blocks = nn.ModuleList([
            FNOBlock(width, modes, corners, init, norm, bypass_kernel, mlp_ratio, residual)
            for _ in range(n_layers)])
        self.proj = nn.Sequential(nn.Conv2d(width, proj_hidden, 1), nn.GELU(),
                                  nn.Conv2d(proj_hidden, out_ch, 1))

    def _prep(self, x):
        if self.input_norm == 'max':
            return x / x.amax((1, 2, 3), keepdim=True).clamp_min(1e-6)
        if self.input_norm == 'meanstd':
            return (x - x.mean((1, 2, 3), True)) / x.std((1, 2, 3), keepdim=True).clamp_min(1e-6)
        return x

    def forward(self, x, return_logits=False):
        x = self._prep(x)
        B, C, H, W = x.shape
        if self.use_grid:
            gy = torch.linspace(0, 1, H, device=x.device, dtype=x.dtype)
            gx = torch.linspace(0, 1, W, device=x.device, dtype=x.dtype)
            yy, xx = torch.meshgrid(gy, gx, indexing='ij')
            x = torch.cat([x, yy[None, None].expand(B, 1, H, W),
                           xx[None, None].expand(B, 1, H, W)], 1)
        x = self.lift(x)
        if self.padding > 0:
            ph, pw = int(round(H * self.padding)), int(round(W * self.padding))
            x = F.pad(x, (0, pw, 0, ph))
        for b in self.blocks:
            x = b(x)
        if self.padding > 0:
            x = x[..., :H, :W]
        z = self.proj(x)
        return z if (return_logits or not self.out_sigmoid) else torch.sigmoid(z)

# Define the LPIPS + MSE loss function.

LPIPS_W_START = 0.1
LPIPS_W_END = 0.9


def lpips_mse_weights(ep, total):
    """LPIPS / MSE weights for a 0-indexed epoch (the schedule lives in one place)."""
    if total <= 1:
        w = LPIPS_W_START
    else:
        f = min(max(int(ep), 0), total - 1) / (total - 1)
        w = LPIPS_W_START + (LPIPS_W_END - LPIPS_W_START) * f
    return float(w), float(1.0 - w)


class LPIPSLoss(nn.Module):
    """Differentiable LPIPS (AlexNet backbone) for use as a training loss.

    The torchmetrics module is preferred because it is already a dependency and its
    forward pass propagates gradients; the reference `lpips` package is the fallback.
    Whichever backend is selected is *gradient-checked* at construction, so a silent
    "loss that cannot train" is impossible. A dedicated instance is used here: the LPIPS
    object in the metric section must stay free of the loss's state.
    """

    def __init__(self, device='cuda'):
        super().__init__()
        self.backend, self.net, errs = None, None, []
        try:
            self.net = LearnedPerceptualImagePatchSimilarity(net_type='alex',
                                                             normalize=True).to(device)
            self.backend = 'torchmetrics'
        except Exception as e:
            errs.append(f"torchmetrics: {type(e).__name__} {e}")
        if self.net is None:
            try:
                import lpips as _lpips_pkg
                self.net = _lpips_pkg.LPIPS(net='alex').to(device).eval()
                self.backend = 'lpips'
            except Exception as e:
                errs.append(f"lpips package: {type(e).__name__} {e}")
        assert self.net is not None, "no LPIPS backend available: " + " | ".join(errs)
        for p in self.net.parameters():
            p.requires_grad_(False)

        x = torch.rand(1, 3, 64, 64, device=device, requires_grad=True)
        g = torch.autograd.grad(self(x, torch.rand(1, 3, 64, 64, device=device)), x)[0]
        assert torch.isfinite(g).all() and float(g.abs().sum()) > 0, \
            f"LPIPS backend '{self.backend}' does not propagate gradients"

    def forward(self, pred, target):
        if self.backend == 'lpips':
            return self.net(pred * 2 - 1, target * 2 - 1).mean()   # package expects [-1, 1]
        v = self.net(pred.clamp(0, 1), target.clamp(0, 1))         # torchmetrics: [0, 1]
        self.net.reset()                                           # keep no metric state
        return v


class LpipsMseLoss(nn.Module):
    """w_lpips(ep) * LPIPS + w_mse(ep) * MSE.

    w_lpips starts at LPIPS_W_START in the first epoch and ramps linearly to LPIPS_W_END in
    the final epoch; w_mse = 1 - w_lpips. set_epoch() must be called once per epoch (the
    training loop does this and prints both weights).
    """

    def __init__(self, device='cuda', epochs=50):
        super().__init__()
        self.lpips = LPIPSLoss(device)
        self.mse = nn.MSELoss()
        self.epochs = epochs
        self.w_lpips, self.w_mse = lpips_mse_weights(0, epochs)

    def set_epoch(self, ep):
        self.w_lpips, self.w_mse = lpips_mse_weights(ep, self.epochs)
        return self.w_lpips, self.w_mse

    def forward(self, pred, target):
        return self.w_lpips * self.lpips(pred, target) + self.w_mse * self.mse(pred, target)


# Training loop

in_ch = 3
out_ch = 3

model = FNO2d(
    modes=24,
    width=24,
    n_layers=6,
    corners=2,
    init='sqrt_cin',
    norm='group',
    input_norm='none',
    use_grid=True,
    padding=0.0,
    mlp_ratio=1.0,
    residual=False,
    proj_hidden=128,
    in_ch=in_ch,
    out_ch=out_ch
).to(device)

param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {param_count:,}")

assert param_count == 7_977_443, f"expected 7,977,443 parameters, got {param_count:,}"

epochs = 50
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

criterion = LpipsMseLoss(device=device, epochs=epochs).to(device)
print(f"loss: LPIPS({criterion.lpips.backend}) + MSE, weights ramped per epoch "
      f"({LPIPS_W_START:.2f} -> {LPIPS_W_END:.2f} on LPIPS)")

# Metrics
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex',
                                                     normalize=True).to(device)

history = {'train_loss': [], 'val_psnr': [], 'val_ssim': [],
           'val_lpips': [], 'val_mse': []}

best_val_psnr = 0.0

print(f"Training on {device} for {epochs} epochs...")
print("="*60)

for epoch in range(epochs):
    w_lpips, w_mse = criterion.set_epoch(epoch)
    model.train()
    train_loss = 0.0

    for batch_diffusers, batch_ground_truths in train_loader:
        batch_diffusers = batch_diffusers.to(device)
        batch_ground_truths = batch_ground_truths.to(device)

        optimizer.zero_grad()
        preds = model(batch_diffusers)
        loss = criterion(preds, batch_ground_truths)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    history['train_loss'].append(avg_train_loss)

    model.eval()
    epoch_psnr = 0.0
    epoch_ssim = 0.0
    epoch_lpips = 0.0
    epoch_mse = 0.0

    with torch.no_grad():
        for val_diffusers, val_ground_truths in val_loader:
            val_diffusers = val_diffusers.to(device)
            val_ground_truths = val_ground_truths.to(device)
            val_preds = torch.clamp(model(val_diffusers), 0.0, 1.0)
            epoch_psnr += psnr_metric(val_preds, val_ground_truths).item()
            epoch_ssim += ssim_metric(val_preds, val_ground_truths).item()
            epoch_lpips += lpips_metric(val_preds, val_ground_truths).item()
            epoch_mse += F.mse_loss(val_preds, val_ground_truths).item()
    lpips_metric.reset()

    avg_val_psnr = epoch_psnr / len(val_loader)
    avg_val_ssim = epoch_ssim / len(val_loader)
    avg_val_lpips = epoch_lpips / len(val_loader)
    avg_val_mse = epoch_mse / len(val_loader)
    history['val_psnr'].append(avg_val_psnr)
    history['val_ssim'].append(avg_val_ssim)
    history['val_lpips'].append(avg_val_lpips)
    history['val_mse'].append(avg_val_mse)

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch [{epoch+1:2d}/{epochs}] | Loss: {avg_train_loss:.4f} | w_lpips: {w_lpips:.2f} | PSNR: {avg_val_psnr:.2f} dB | SSIM: {avg_val_ssim:.4f} | LPIPS: {avg_val_lpips:.4f} | MSE: {avg_val_mse:.5f} | LR: {current_lr:.6f}")

    if avg_val_psnr > best_val_psnr:
        best_val_psnr = avg_val_psnr
        torch.save(model.state_dict(), 'FNO_best.pth')

model.load_state_dict(torch.load('FNO_best.pth'))
print(f"\nTraining finished. Best Val PSNR: {best_val_psnr:.2f} dB")
torch.save(model.state_dict(), 'FNO_best.pth')
