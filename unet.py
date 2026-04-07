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


# Load the dataset
# dataset = ...

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

# Define the U-Net model.

BN_EPS = 1e-4

class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=1):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class StackEncoder(nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=3):
        super(StackEncoder, self).__init__()
        padding = (kernel_size - 1) // 2
        self.encode = nn.Sequential(
            ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x):
        x = self.encode(x)
        x_small = F.max_pool2d(x, kernel_size=2, stride=2)
        return x, x_small

class StackDecoder(nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3):
        super(StackDecoder, self).__init__()
        padding = (kernel_size - 1) // 2

        self.decode = nn.Sequential(
            ConvBnRelu2d(x_big_channels + x_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x, down_tensor):
        _, channels, height, width = down_tensor.size()
        x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=False)
        x = torch.cat([x, down_tensor], 1)
        x = self.decode(x)
        return x

class UNet128(nn.Module):
    def __init__(self, in_shape=(3, 128, 128)):
        super(UNet128, self).__init__()

        self.down1 = StackEncoder(3, 64, kernel_size=3)      
        self.down2 = StackEncoder(64, 128, kernel_size=3)    
        self.down3 = StackEncoder(128, 256, kernel_size=3)   

        self.dropout = nn.Dropout2d(p=0.5)

        self.center = nn.Sequential(
            ConvBnRelu2d(256, 512, kernel_size=3, padding=1),
            ConvBnRelu2d(512, 512, kernel_size=3, padding=1),
        )

        self.up3 = StackDecoder(512, 256, 256, kernel_size=3)  
        self.up2 = StackDecoder(256, 128, 128, kernel_size=3)  
        self.up1 = StackDecoder(128, 64, 64, kernel_size=3)    

        self.classify = nn.Conv2d(64, 3, kernel_size=1, bias=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = x
        down1, out = self.down1(out)  
        down2, out = self.down2(out)   
        down3, out = self.down3(out)   

        out = self.dropout(out)
        out = self.center(out)         

        out = self.up3(out, down3)     
        out = self.up2(out, down2)     
        out = self.up1(out, down1)     

        out = self.classify(out)      
        out = self.sigmoid(out)
        return out


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
 
model = UNet128().to(device)
 
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
        torch.save(model.state_dict(), 'UNet128_best.pth')
 
model.load_state_dict(torch.load('UNet128_best.pth'))
print(f"\nTraining finished. Best Val PSNR: {best_val_psnr:.2f} dB")
torch.save(model.state_dict(), 'UNet128_best.pth')