import os
import sys
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Run configuration ──────────────────────────────────────────────────────────
RUN = sys.argv[1] if len(sys.argv) > 1 else "1"
print(f"Evaluating Run: {RUN}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = f"{BASE_DIR}/data/test"

# ── Sharpness Metrics ──────────────────────────────────────────────────────────
def _to_gray_numpy(img_np: np.ndarray) -> np.ndarray:
    """Convert (H,W,3) float32 RGB in [0,1] to (H,W) grayscale using luminance weights."""
    return (0.2989 * img_np[:, :, 0] +
            0.5870 * img_np[:, :, 1] +
            0.1140 * img_np[:, :, 2]).astype(np.float32)

def compute_vol(img_np: np.ndarray) -> float:
    """Variance of Laplacian — measures sharpness via second-order edges."""
    gray = _to_gray_numpy(img_np)
    h, w = gray.shape
    t = torch.from_numpy(gray).float().view(1, 1, h, w)
    lap_kernel = torch.tensor([[0,  1, 0],
                                [1, -4, 1],
                                [0,  1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
    lap_map = F.conv2d(t, lap_kernel, padding=1).squeeze().numpy()
    return float(lap_map.var())

def compute_tenengrad(img_np: np.ndarray) -> float:
    """Tenengrad criterion — measures sharpness via Sobel gradient magnitude variance."""
    gray = _to_gray_numpy(img_np)
    h, w = gray.shape
    t = torch.from_numpy(gray).float().view(1, 1, h, w)
    Sx = torch.tensor([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    Sy = torch.tensor([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)
    Gx = F.conv2d(t, Sx, padding=1).squeeze().numpy()
    Gy = F.conv2d(t, Sy, padding=1).squeeze().numpy()
    M = np.sqrt(Gx ** 2 + Gy ** 2)
    return float(M.var())

# ── ALL RUNS SUMMARY ──────────────────────────────────────────────────────────
if RUN == "all":
    all_psnr_ae,  all_ssim_ae  = [], []
    all_psnr_gan, all_ssim_gan = [], []
    all_vol_ae,   all_ten_ae   = [], []
    all_vol_gan,  all_ten_gan  = [], []

    for r in [1, 2, 3]:
        metrics_path = f"{BASE_DIR}/results/run{r}/metrics.txt"
        if not os.path.exists(metrics_path):
            print(f"WARNING: results/run{r}/metrics.txt not found — run 'python evaluate.py {r}' first.")
            continue
        with open(metrics_path) as f:
            lines = f.readlines()
        # format: Model,PSNR,SSIM,VoL,TEN
        ae_vals  = lines[1].strip().split(",")
        gan_vals = lines[2].strip().split(",")
        all_psnr_ae.append(float(ae_vals[1]))
        all_ssim_ae.append(float(ae_vals[2]))
        all_vol_ae.append(float(ae_vals[3]))
        all_ten_ae.append(float(ae_vals[4]))
        all_psnr_gan.append(float(gan_vals[1]))
        all_ssim_gan.append(float(gan_vals[2]))
        all_vol_gan.append(float(gan_vals[3]))
        all_ten_gan.append(float(gan_vals[4]))

    if len(all_psnr_ae) == 0:
        print("No runs found. Run evaluate.py 1, 2, 3 first.")
        sys.exit(1)

    print("\n" + "="*85)
    print(f"{'Model':<20} {'PSNR mean+-std':>20} {'SSIM mean+-std':>20} {'VoL mean+-std':>18} {'TEN mean+-std':>18}")
    print("="*85)
    print(f"{'Autoencoder':<20} "
          f"{np.mean(all_psnr_ae):>8.4f}+/-{np.std(all_psnr_ae):.4f}   "
          f"{np.mean(all_ssim_ae):>8.4f}+/-{np.std(all_ssim_ae):.4f}   "
          f"{np.mean(all_vol_ae):>8.4f}+/-{np.std(all_vol_ae):.4f}   "
          f"{np.mean(all_ten_ae):>8.4f}+/-{np.std(all_ten_ae):.4f}")
    print(f"{'GAN Enhanced':<20} "
          f"{np.mean(all_psnr_gan):>8.4f}+/-{np.std(all_psnr_gan):.4f}   "
          f"{np.mean(all_ssim_gan):>8.4f}+/-{np.std(all_ssim_gan):.4f}   "
          f"{np.mean(all_vol_gan):>8.4f}+/-{np.std(all_vol_gan):.4f}   "
          f"{np.mean(all_ten_gan):>8.4f}+/-{np.std(all_ten_gan):.4f}")
    print("="*85)

    summary_path = f"{BASE_DIR}/results/summary.txt"
    os.makedirs(f"{BASE_DIR}/results", exist_ok=True)
    with open(summary_path, "w") as f:
        f.write("Model,PSNR_mean,PSNR_std,SSIM_mean,SSIM_std,VoL_mean,VoL_std,TEN_mean,TEN_std\n")
        f.write(f"Autoencoder,"
                f"{np.mean(all_psnr_ae):.4f},{np.std(all_psnr_ae):.4f},"
                f"{np.mean(all_ssim_ae):.4f},{np.std(all_ssim_ae):.4f},"
                f"{np.mean(all_vol_ae):.4f},{np.std(all_vol_ae):.4f},"
                f"{np.mean(all_ten_ae):.4f},{np.std(all_ten_ae):.4f}\n")
        f.write(f"GAN Enhanced,"
                f"{np.mean(all_psnr_gan):.4f},{np.std(all_psnr_gan):.4f},"
                f"{np.mean(all_ssim_gan):.4f},{np.std(all_ssim_gan):.4f},"
                f"{np.mean(all_vol_gan):.4f},{np.std(all_vol_gan):.4f},"
                f"{np.mean(all_ten_gan):.4f},{np.std(all_ten_gan):.4f}\n")
    print(f"\nSaved summary -> {summary_path}")
    sys.exit(0)

# ── PER-RUN EVALUATION ────────────────────────────────────────────────────────
RESULTS_DIR = f"{BASE_DIR}/results/run{RUN}"
WEIGHTS_DIR = f"{BASE_DIR}/weights/run{RUN}"
os.makedirs(RESULTS_DIR, exist_ok=True)

IMAGE_SIZE = 100
LATENT_DIM = 128
BATCH_SIZE = 64
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Dataset ────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

class MalariaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform   = transform
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

test_dataset = MalariaDataset(TEST_DIR, transform=transform)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
print(f"Test images: {len(test_dataset)}")

# ── Autoencoder ────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32,  kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(128 * 13 * 13, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 13 * 13)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3,  kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 128, 13, 13)
        x = self.deconv(x)
        x = x[:, :, 2:102, 2:102]
        return x


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ── Generator ──────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(*[ResBlock(64) for _ in range(8)])
        self.exit = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        entry = self.entry(x)
        res   = self.res_blocks(entry)
        return self.exit(res + entry)


# ── Load weights ───────────────────────────────────────────────────────────────
autoencoder = Autoencoder().to(DEVICE)
autoencoder.load_state_dict(torch.load(
    os.path.join(WEIGHTS_DIR, "autoencoder.pt"),
    map_location=DEVICE, weights_only=True
))
autoencoder.eval()
print(f"Loaded {WEIGHTS_DIR}/autoencoder.pt")

generator = Generator().to(DEVICE)
generator.load_state_dict(torch.load(
    os.path.join(WEIGHTS_DIR, "generator.pt"),
    map_location=DEVICE, weights_only=True
))
generator.eval()
print(f"Loaded {WEIGHTS_DIR}/generator.pt")

# ── Helpers ────────────────────────────────────────────────────────────────────
def compute_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).mean()
    return 10 * math.log10(1.0 / mse) if mse > 0 else 100.0

def compute_ssim(img1, img2):
    return ssim(img1, img2, data_range=1.0, channel_axis=2)

# ── Inference on full test set ─────────────────────────────────────────────────
print("Running inference on test set...")
psnr_ae_list,  ssim_ae_list  = [], []
psnr_gan_list, ssim_gan_list = [], []
vol_ae_list,   ten_ae_list   = [], []
vol_gan_list,  ten_gan_list  = [], []

with torch.no_grad():
    for imgs in test_loader:
        imgs    = imgs.to(DEVICE)
        ae_out  = autoencoder(imgs)
        gan_out = generator(ae_out)

        for i in range(imgs.size(0)):
            orig_np = imgs[i].permute(1,2,0).cpu().numpy()
            ae_np   = ae_out[i].permute(1,2,0).cpu().numpy()
            gan_np  = gan_out[i].permute(1,2,0).cpu().numpy()

            psnr_ae_list.append(compute_psnr(orig_np, ae_np))
            ssim_ae_list.append(compute_ssim(orig_np, ae_np))
            vol_ae_list.append(compute_vol(ae_np))
            ten_ae_list.append(compute_tenengrad(ae_np))

            psnr_gan_list.append(compute_psnr(orig_np, gan_np))
            ssim_gan_list.append(compute_ssim(orig_np, gan_np))
            vol_gan_list.append(compute_vol(gan_np))
            ten_gan_list.append(compute_tenengrad(gan_np))

mean_psnr_ae  = sum(psnr_ae_list)  / len(psnr_ae_list)
mean_ssim_ae  = sum(ssim_ae_list)  / len(ssim_ae_list)
mean_vol_ae   = sum(vol_ae_list)   / len(vol_ae_list)
mean_ten_ae   = sum(ten_ae_list)   / len(ten_ae_list)

mean_psnr_gan = sum(psnr_gan_list) / len(psnr_gan_list)
mean_ssim_gan = sum(ssim_gan_list) / len(ssim_gan_list)
mean_vol_gan  = sum(vol_gan_list)  / len(vol_gan_list)
mean_ten_gan  = sum(ten_gan_list)  / len(ten_gan_list)

# ── Print results ──────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"{'Model':<20} {'PSNR (dB)':>12} {'SSIM':>10} {'VoL':>10} {'TEN':>10}")
print("="*70)
print(f"{'Autoencoder':<20} {mean_psnr_ae:>12.4f} {mean_ssim_ae:>10.4f} {mean_vol_ae:>10.4f} {mean_ten_ae:>10.4f}")
print(f"{'GAN Enhanced':<20} {mean_psnr_gan:>12.4f} {mean_ssim_gan:>10.4f} {mean_vol_gan:>10.4f} {mean_ten_gan:>10.4f}")
print("="*70 + "\n")

# ── Save metrics.txt ───────────────────────────────────────────────────────────
metrics_path = os.path.join(RESULTS_DIR, "metrics.txt")
with open(metrics_path, "w") as f:
    f.write("Model,PSNR,SSIM,VoL,TEN\n")
    f.write(f"Autoencoder,{mean_psnr_ae:.4f},{mean_ssim_ae:.4f},{mean_vol_ae:.4f},{mean_ten_ae:.4f}\n")
    f.write(f"GAN Enhanced,{mean_psnr_gan:.4f},{mean_ssim_gan:.4f},{mean_vol_gan:.4f},{mean_ten_gan:.4f}\n")
print(f"Saved metrics -> {metrics_path}")

# ── Save inference figure (Original vs AE vs GAN, 5 images) ───────────────────
random.seed(42)
indices = random.sample(range(len(test_dataset)), 5)

fig = plt.figure(figsize=(15, 10))
gs  = gridspec.GridSpec(
    6, 5,
    height_ratios=[0.12, 1, 0.18, 1, 0.18, 1],
    hspace=0.3, wspace=0.05
)
fig.suptitle(f"Inference Results (Run {RUN}): Original vs AE vs GAN", fontsize=13, y=1.005)

with torch.no_grad():
    for col, idx in enumerate(indices):
        img     = test_dataset[idx].unsqueeze(0).to(DEVICE)
        ae_out  = autoencoder(img)
        gan_out = generator(ae_out)

        orig_np  = img.squeeze().permute(1,2,0).cpu().numpy()
        ae_np    = ae_out.squeeze().permute(1,2,0).cpu().numpy()
        gan_np   = gan_out.squeeze().permute(1,2,0).cpu().numpy()

        psnr_ae  = compute_psnr(orig_np, ae_np)
        ssim_ae  = compute_ssim(orig_np, ae_np)
        psnr_gan = compute_psnr(orig_np, gan_np)
        ssim_gan = compute_ssim(orig_np, gan_np)

        ax = fig.add_subplot(gs[0, col])
        ax.text(0.5, 0.5, "Original", ha='center', va='center', fontsize=10, fontweight='bold')
        ax.axis('off')

        ax = fig.add_subplot(gs[1, col])
        ax.imshow(orig_np, vmin=0, vmax=1)
        ax.axis('off')

        ax = fig.add_subplot(gs[2, col])
        ax.text(0.5, 0.5, "AE Reconstructed", ha='center', va='center', fontsize=10, fontweight='bold')
        ax.axis('off')

        ax = fig.add_subplot(gs[3, col])
        ax.imshow(ae_np, vmin=0, vmax=1)
        ax.set_xlabel(f"PSNR: {psnr_ae:.2f} dB | SSIM: {ssim_ae:.4f}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = fig.add_subplot(gs[4, col])
        ax.text(0.5, 0.5, "GAN Enhanced", ha='center', va='center', fontsize=10, fontweight='bold')
        ax.axis('off')

        ax = fig.add_subplot(gs[5, col])
        ax.imshow(gan_np, vmin=0, vmax=1)
        ax.set_xlabel(f"PSNR: {psnr_gan:.2f} dB | SSIM: {ssim_gan:.4f}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

save_path = os.path.join(RESULTS_DIR, "inference_results.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved inference figure -> {save_path}")