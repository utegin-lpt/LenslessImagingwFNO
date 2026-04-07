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
 
transform = transforms.Compose([
    transforms.Resize((128, 128)),
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
 
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
 
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
 
    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)
 
    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
 
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
 
        m1 = min(self.modes1, x_ft.size(-2) // 2)
        m2 = min(self.modes2, x_ft.size(-1))
 
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
        )
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
        )
 
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
 
 
class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, n_layers=6, in_channels=3, out_channels=3):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
 
        self.p = nn.Linear(in_channels + 2, self.width)
 
        # Build n_layers spectral conv + 1x1 conv pairs
        self.spectral_convs = nn.ModuleList([
            SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            for _ in range(n_layers)
        ])
        self.w_convs = nn.ModuleList([
            nn.Conv2d(self.width, self.width, 1)
            for _ in range(n_layers)
        ])
 
        # Project back
        self.q = nn.Linear(self.width, 128)
        self.q2 = nn.Linear(128, out_channels)
 
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x, device=device).reshape(1, size_x, 1, 1).expand(batchsize, -1, size_y, -1)
        gridy = torch.linspace(0, 1, size_y, device=device).reshape(1, 1, size_y, 1).expand(batchsize, size_x, -1, -1)
        return torch.cat((gridx, gridy), dim=-1)
 
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
 
        # Dynamic padding
        pad_h = max(1, int(round(x.shape[-2] * 0.125)))
        pad_w = max(1, int(round(x.shape[-1] * 0.125)))
        x = F.pad(x, [0, pad_w, 0, pad_h])
 
        # Spectral layers with residual connections every 2 layers
        for i in range(self.n_layers):
            if i % 2 == 0:
                x_res = x
 
            x1 = self.spectral_convs[i](x)
            x2 = self.w_convs[i](x)
            x = x1 + x2
 
            if i % 2 == 1:
                x = x + x_res
 
            # GELU on all but the last layer
            if i < self.n_layers - 1:
                x = F.gelu(x)
 
        if pad_h > 0:
            x = x[..., :-pad_h, :]
        if pad_w > 0:
            x = x[..., :, :-pad_w]
 
        # Project to img space
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.q(x))
        x = self.q2(x)
        x = x.permute(0, 3, 1, 2)
        return x
 
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
 
in_ch = 3
out_ch = 3
 
model = FNO2d(
    modes1=24,
    modes2=24,
    width=34,
    n_layers=6,
    in_channels=in_ch,
    out_channels=out_ch
).to(device)
 
param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {param_count:,}")
 
epochs = 50
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
 
criterion = CombinedLoss(device=device, w_ssim_l1=1.0, w_perceptual=0.05)
 
# Metrics
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
 
history = {'train_loss': [], 'val_psnr': [], 'val_ssim': []}
 
best_val_psnr = 0.0
 
print(f"Training on {device} for {epochs} epochs...")
print("="*60)
 
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
        torch.save(model.state_dict(), 'FNO_best.pth')
 
model.load_state_dict(torch.load('FNO_best.pth'))
print(f"\nTraining finished. Best Val PSNR: {best_val_psnr:.2f} dB")
torch.save(model.state_dict(), 'FNO_best.pth')