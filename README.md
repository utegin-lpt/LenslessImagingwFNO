# LenslessImagingwFNO

Official code for the manuscript **"Resolution-Agnostic Lensless Imaging via Fourier Neural Operators"** by Ekec et al.

We use a Fourier Neural Operator (FNO) to reconstruct images from a DiffuserCam lensless camera. Because the FNO learns a mapping between continuous function spaces, a single network trained at 128×128 reconstructs measurements at 256×256 and 512×512 *without retraining*.



## Highlights

- **+2.03 dB PSNR / +0.11 SSIM** over a parameter-matched U-Net baseline at 128×128.
- **Resolution-agnostic inference**: <1 dB PSNR drop when evaluated at 2× and 4× finer discretizations than training.
- **Lower training cost**: 93 s/epoch and 3.9 GB VRAM vs. 102 s/epoch and 5.2 GB for U-Net.
- First application of neural operators to optical lensless imaging.

Tested with PyTorch 2.10.0 and CUDA 11.8 on a single NVIDIA A100.

## Project Structure

* `FNO.py` — Defines the Fourier Neural Operator (`SpectralConv2d`, `FNO2d`), the combined SSIM-L1 + VGG perceptual loss, the paired DiffuserCam dataloader, and the full training loop. Trains the FNO at 128×128 and saves the best checkpoint as `FNO_best.pth`.
* `unet.py` — Defines the U-Net baseline (`UNet128`) following Ronneberger et al., with 3 encoder and 3 decoder stages plus skip connections. Shares the same dataloader, loss, optimizer, and training loop as `FNO.py` for a fair comparison. Saves the best checkpoint as `UNet128_best.pth`.
* `eval_multires.py` — Loads the trained FNO and evaluates it at 128, 256, and 512 to reproduce the resolution-agnostic results in Table 2 and Figure 4 of the paper.


## Dataset

We use the **MIRFlickr-25k** dataset displayed on a tablet and captured simultaneously by a lensed reference camera and a DiffuserCam (bare sensor with double-sided Scotch tape diffuser, ~3 mm sensor distance). 25,000 image pairs are split into 23,000 training, 1000 validation and 1,000 test pairs; resized to 512×512.

For users who do not have access to our hardware, the publicly available DiffuserCam dataset from the Waller Lab can be used as a drop-in replacement: <https://waller-lab.github.io/LenslessLearning/>.

