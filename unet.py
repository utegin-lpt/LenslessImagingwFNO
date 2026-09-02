import os
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

# Define the U-Net model.
# depth=3, base=64, k=3 reproduces the U-Net baseline *exactly*: 7,785,859 parameters
# (asserted below). Dilation changes the receptive field without changing a single
# weight, so the dilated variant used here has the identical parameter count. The
# decoder interpolates to each skip's shape, so the parameter count and the layer
# layout do not depend on the input size either.

BN_EPS = 1e-4

class ConvBnAct(nn.Module):
    def __init__(self, cin, cout, k=3, dilation=1, padding_mode='zeros'):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.conv = nn.Conv2d(cin, cout, k, padding=pad, dilation=dilation,
                              bias=False, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(cout, eps=BN_EPS)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BlurPool(nn.Module):
    """Anti-aliased downsampling (binomial 3x3 blur then stride-2)."""
    def __init__(self, ch):
        super().__init__()
        k = torch.tensor([1., 2., 1.])
        k = (k[:, None] * k[None, :]) / 16.0
        self.register_buffer('k', k[None, None].repeat(ch, 1, 1, 1))
        self.ch = ch

    def forward(self, x):
        return F.conv2d(x, self.k.to(x.dtype), stride=2, padding=1, groups=self.ch)


def make_pool(kind, ch):
    if kind == 'max':    return nn.MaxPool2d(2, 2)
    if kind == 'avg':    return nn.AvgPool2d(2, 2)
    if kind == 'blur':   return BlurPool(ch)
    if kind == 'stride': return nn.Conv2d(ch, ch, 3, stride=2, padding=1, bias=False)
    raise ValueError(kind)


class Decoder(nn.Module):
    def __init__(self, c_big, c_skip, c_out, k=3, padding_mode='zeros', dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnAct(c_big + c_skip, c_out, k, dilation, padding_mode),
            ConvBnAct(c_out, c_out, k, dilation, padding_mode),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.block(torch.cat([x, skip], 1))


def make_bottleneck(kind, ch):
    if kind in (None, 'none'):
        return nn.Identity()
    raise ValueError(f"bottleneck='{kind}' is not reproduced here; this baseline uses 'none'")


class UNet(nn.Module):
    """
    depth            : number of pool/unpool levels (0 = no downsampling at all)
    base             : channels at the finest level; doubles each level, capped at ch_max
    k                : conv kernel size everywhere
    dilation         : dilation applied to EVERY encoder/decoder conv
    center_dilations : dilation of the bottleneck convs
                       (both dilation knobs change the RF at IDENTICAL parameter count)
    pool             : 'max' | 'avg' | 'blur' | 'stride'
    input_maxnorm    : per-image max normalisation of the input (off = the baseline's
                       behaviour, i.e. the raw dark measurement; on = the same
                       preprocessing the FNO applies internally)
    """
    def __init__(self, depth=3, base=64, k=3, ch_max=512, center_convs=2,
                 dilation=1, center_dilations=(1, 1), pool='max', padding_mode='zeros',
                 bottleneck='none', dropout=0.5, in_ch=3, out_ch=3,
                 out_sigmoid=True, input_maxnorm=False):
        super().__init__()
        self.depth, self.out_sigmoid = depth, out_sigmoid
        self.input_maxnorm = input_maxnorm
        chs = [min(base * 2 ** i, ch_max) for i in range(depth)]
        c_center = min(base * 2 ** depth, ch_max)

        self.encs, self.pools = nn.ModuleList(), nn.ModuleList()
        cin = in_ch
        for i in range(depth):
            self.encs.append(nn.Sequential(ConvBnAct(cin, chs[i], k, dilation, padding_mode),
                                           ConvBnAct(chs[i], chs[i], k, dilation, padding_mode)))
            self.pools.append(make_pool(pool, chs[i]))
            cin = chs[i]

        dil = list(center_dilations)
        while len(dil) < center_convs:
            dil.append(dil[-1])
        layers, c = [], cin
        for j in range(center_convs):
            layers.append(ConvBnAct(c, c_center, k, dil[j], padding_mode))
            c = c_center
        self.center = nn.Sequential(*layers)
        self.bneck = make_bottleneck(bottleneck, c_center)
        self.drop = nn.Dropout2d(dropout) if (depth > 0 and dropout > 0) else nn.Identity()

        self.decs = nn.ModuleList()
        cin = c_center
        for i in reversed(range(depth)):
            self.decs.append(Decoder(cin, chs[i], chs[i], k, padding_mode, dilation))
            cin = chs[i]
        self.head = nn.Conv2d(cin, out_ch, 1, bias=True)

    def forward(self, x, return_logits=False):
        if self.input_maxnorm:
            x = x / x.amax((1, 2, 3), keepdim=True).clamp_min(1e-6)
        feats = []
        for enc, pool in zip(self.encs, self.pools):
            x = enc(x); feats.append(x); x = pool(x)
        x = self.bneck(self.center(self.drop(x)))
        for dec, skip in zip(self.decs, reversed(feats)):
            x = dec(x, skip)
        z = self.head(x)
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


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
    out_sigmoid=True,
    input_maxnorm=False
).to(device)

param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {param_count:,}")

assert param_count == 7_785_859, f"expected 7,785,859 parameters, got {param_count:,}"

epochs = 50
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

criterion = LpipsMseLoss(device=device, epochs=epochs).to(device)
print(f"loss: LPIPS({criterion.lpips.backend}) + MSE, weights ramped per epoch "
      f"({LPIPS_W_START:.2f} -> {LPIPS_W_END:.2f} on LPIPS)")

psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex',
                                                     normalize=True).to(device)

history = {'train_loss': [], 'val_psnr': [], 'val_ssim': [],
           'val_lpips': [], 'val_mse': []}

best_val_psnr = 0.0

print(f"Training will start for {epochs} epochs...")

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
        torch.save(model.state_dict(), 'UNet_best.pth')

model.load_state_dict(torch.load('UNet_best.pth'))
print(f"\nTraining finished. Best Val PSNR: {best_val_psnr:.2f} dB")
torch.save(model.state_dict(), 'UNet_best.pth')
