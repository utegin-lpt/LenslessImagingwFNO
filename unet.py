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
from torchvision.models import vgg16, VGG16_Weights
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio


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
# depth=3, base=64, k=3 reproduces the U-Net baseline *exactly*: 7,785,859 parameters.
# Dilation changes the receptive field without changing a single
# weight, so the dilated variant used here has the identical parameter count. 

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
                 bottleneck='none', dropout=0.5, in_channels=3, out_channels=3,
                 out_sigmoid=True, input_maxnorm=False):
        super().__init__()
        self.depth, self.out_sigmoid = depth, out_sigmoid
        self.input_maxnorm = input_maxnorm
        chs = [min(base * 2 ** i, ch_max) for i in range(depth)]
        c_center = min(base * 2 ** depth, ch_max)

        self.encs, self.pools = nn.ModuleList(), nn.ModuleList()
        cin = in_channels
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
        self.head = nn.Conv2d(cin, out_channels, 1, bias=True)

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


# Define the combined loss function.

class SSIM_L1_Loss(nn.Module):
    def __init__(self, alpha=0.5, data_range=1.0, device='cuda'):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        ssim_loss = 1.0 - self.ssim(pred, target)
        return self.alpha * l1_loss + (1.0 - self.alpha) * ssim_loss


class PerceptualLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.feature_extractor = vgg[:16].eval().to(device)
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        if pred.shape[-1] < 64 or pred.shape[-2] < 64:
            pred = F.interpolate(pred, size=(64, 64), mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(64, 64), mode='bilinear', align_corners=False)
        pred_features = self.feature_extractor(self.normalize(pred))
        target_features = self.feature_extractor(self.normalize(target))
        return F.l1_loss(pred_features, target_features)


class CombinedLoss(nn.Module):
    def __init__(self, device='cuda', w_ssim_l1=1.0, w_perceptual=0.05):
        super().__init__()
        self.ssim_l1 = SSIM_L1_Loss(alpha=0.5, data_range=1.0, device=device)
        self.perceptual = PerceptualLoss(device=device)
        self.w_ssim_l1 = w_ssim_l1
        self.w_perceptual = w_perceptual

    def forward(self, pred, target):
        loss_sl = self.ssim_l1(pred, target)
        loss_p = self.perceptual(pred, target)
        return self.w_ssim_l1 * loss_sl + self.w_perceptual * loss_p


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

criterion = CombinedLoss(device=device, w_ssim_l1=1.0, w_perceptual=0.05)

psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

history = {'train_loss': [], 'val_psnr': [], 'val_ssim': []}

best_val_psnr = 0.0

print(f"Training will start for {epochs} epochs...")

for epoch in range(epochs):
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

    with torch.no_grad():
        for val_diffusers, val_ground_truths in val_loader:
            val_diffusers = val_diffusers.to(device)
            val_ground_truths = val_ground_truths.to(device)
            val_preds = torch.clamp(model(val_diffusers), 0.0, 1.0)
            epoch_psnr += psnr_metric(val_preds, val_ground_truths).item()
            epoch_ssim += ssim_metric(val_preds, val_ground_truths).item()

    avg_val_psnr = epoch_psnr / len(val_loader)
    avg_val_ssim = epoch_ssim / len(val_loader)
    history['val_psnr'].append(avg_val_psnr)
    history['val_ssim'].append(avg_val_ssim)

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch [{epoch+1:2d}/{epochs}] | Loss: {avg_train_loss:.4f} | PSNR: {avg_val_psnr:.2f} dB | SSIM: {avg_val_ssim:.4f} | LR: {current_lr:.6f}")

    if avg_val_psnr > best_val_psnr:
        best_val_psnr = avg_val_psnr
        torch.save(model.state_dict(), 'UNet_best.pth')

model.load_state_dict(torch.load('UNet_best.pth'))
print(f"\nTraining finished. Best Val PSNR: {best_val_psnr:.2f} dB")
torch.save(model.state_dict(), 'UNet_best.pth')
