# LenslessImagingwFNO

Official code for **"Resolution-Agnostic Lensless Imaging via Fourier Neural Operators"** (Ekec et al.). 

We use a Fourier Neural Operator (FNO) to reconstruct DiffuserCam measurements. Because FNOs learn continuous function mappings, a single network trained at 128×128 can reconstruct images at 256×256 and 512×512 *without retraining*.

## Highlights
* **Superior Performance:** +2.03 dB PSNR and +0.11 SSIM over a parameter-matched U-Net at 128×128.
* **Resolution-Agnostic:** <1 dB PSNR drop when evaluating at 2× and 4× finer resolutions.
* **Efficient:** Lower training cost (93s/epoch, 3.9GB VRAM) compared to U-Net (102s/epoch, 5.2GB VRAM).
* **Novelty:** The first application of neural operators to optical lensless imaging.
* **Requirements:** PyTorch 2.10.0, CUDA 11.8. 

## Project Structure
* `FNO.py` — Defines the FNO architecture, custom loss, and full training loop (outputs `FNO_best.pth`).
* `unet.py` — U-Net baseline training script using an identical setup for fair comparison (outputs `UNet128_best.pth`).
* `eval_multires.py` — Evaluates the trained FNO across 128, 256, and 512 resolutions to reproduce paper results.

## Dataset
We use 25,000 paired images (MIRFlickr-25k displayed on a tablet, captured via a lensed camera and a bare-sensor DiffuserCam), resized to 512×512 and split into 23k train / 1k val / 1k test.

The public [Waller Lab DiffuserCam dataset](https://waller-lab.github.io/LenslessLearning/) serves as a drop-in replacement for those without access to our hardware.
