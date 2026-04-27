import os
import sys
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Run configuration ──────────────────────────────────────────────────────────
# Usage: python gan.py <run_number>
# Example: python gan.py 1   (seed=42)
#          python gan.py 2   (seed=43)
#          python gan.py 3   (seed=44)
RUN        = int(sys.argv[1]) if len(sys.argv) > 1 else 1
SEED       = 41 + RUN  # run1=42, run2=43, run3=44

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print(f"Run {RUN} | Seed {SEED}")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = "/home/zn0004/CPE487587_SP26_Final_Project"
TRAIN_DIR   = f"{BASE_DIR}/data/train"
VAL_DIR     = f"{BASE_DIR}/data/val"
TEST_DIR    = f"{BASE_DIR}/data/test"
RESULTS_DIR = f"{BASE_DIR}/results/run{RUN}"
WEIGHTS_DIR = f"{BASE_DIR}/weights/run{RUN}"
LOGS_DIR    = f"{BASE_DIR}/logs/run{RUN}"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,    exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
IMAGE_SIZE         = 100
LATENT_DIM         = 128
BATCH_SIZE         = 64
WARMUP_EPOCHS      = 20
ADVERSARIAL_EPOCHS = 80
TOTAL_EPOCHS       = WARMUP_EPOCHS + ADVERSARIAL_EPOCHS
LR                 = 1e-4
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

train_dataset = MalariaDataset(TRAIN_DIR, transform=transform)
val_dataset   = MalariaDataset(VAL_DIR,   transform=transform)
test_dataset  = MalariaDataset(TEST_DIR,  transform=transform)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

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


# ── Load frozen autoencoder ────────────────────────────────────────────────────
autoencoder = Autoencoder().to(DEVICE)
autoencoder.load_state_dict(torch.load(
    os.path.join(WEIGHTS_DIR, "autoencoder.pt"),
    map_location=DEVICE,
    weights_only=True
))
autoencoder.eval()
for param in autoencoder.parameters():
    param.requires_grad = False
print("Loaded frozen autoencoder.")

# ── Degradation: blur + noise applied to AE output during training ─────────────
def degrade(ae_out):
    # Gaussian blur via avg_pool approximation
    blurred = F.avg_pool2d(ae_out, kernel_size=3, stride=1, padding=1)
    # Add Gaussian noise
    noise   = torch.randn_like(blurred) * 0.05
    return torch.clamp(blurred + noise, 0.0, 1.0)

# ── GAN: Residual Block ────────────────────────────────────────────────────────
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


# ── GAN: Generator (8 ResBlocks + global skip) ────────────────────────────────
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
        out   = self.exit(res + entry)
        return out


# ── GAN: Discriminator (PatchGAN) ─────────────────────────────────────────────
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 1, kernel_size=2, stride=1, padding=0),
        )

    def forward(self, x):
        return self.model(x)


# ── VGG16 Perceptual Loss ──────────────────────────────────────────────────────
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:9]
        self.vgg = vgg.to(DEVICE).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        return self.criterion(self.vgg(pred), self.vgg(target))


# ── Init models ────────────────────────────────────────────────────────────────
generator     = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)
perceptual    = PerceptualLoss()

opt_g = torch.optim.AdamW(generator.parameters(),     lr=LR, betas=(0.5, 0.999))
opt_d = torch.optim.AdamW(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

criterion_l1  = nn.L1Loss()
criterion_adv = nn.BCEWithLogitsLoss()
scaler        = torch.amp.GradScaler('cuda')

# ── Helpers ────────────────────────────────────────────────────────────────────
def compute_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).mean()
    return 10 * math.log10(1.0 / mse) if mse > 0 else 100.0

def compute_ssim(img1, img2):
    return ssim(img1, img2, data_range=1.0, channel_axis=2)

def validate(generator, autoencoder, val_loader):
    generator.eval()
    psnr_ae_list, ssim_ae_list   = [], []
    psnr_gan_list, ssim_gan_list = [], []

    with torch.no_grad():
        for imgs in val_loader:
            imgs    = imgs.to(DEVICE)
            ae_out  = autoencoder(imgs)
            # at inference: no degradation, use clean AE output
            gan_out = generator(ae_out)

            for i in range(imgs.size(0)):
                orig_np = imgs[i].permute(1,2,0).cpu().numpy()
                ae_np   = ae_out[i].permute(1,2,0).cpu().numpy()
                gan_np  = gan_out[i].permute(1,2,0).cpu().numpy()

                psnr_ae_list.append(compute_psnr(orig_np, ae_np))
                ssim_ae_list.append(compute_ssim(orig_np, ae_np))
                psnr_gan_list.append(compute_psnr(orig_np, gan_np))
                ssim_gan_list.append(compute_ssim(orig_np, gan_np))

    return (
        sum(psnr_ae_list)  / len(psnr_ae_list),
        sum(ssim_ae_list)  / len(ssim_ae_list),
        sum(psnr_gan_list) / len(psnr_gan_list),
        sum(ssim_gan_list) / len(ssim_gan_list),
    )

# ── Training ───────────────────────────────────────────────────────────────────
log_path = os.path.join(LOGS_DIR, "gan.log")

with open(log_path, "w") as log_file:
    log_file.write("epoch,phase,train_gen_loss,train_disc_loss,val_psnr_ae,val_ssim_ae,val_psnr_gan,val_ssim_gan\n")

    for epoch in range(1, TOTAL_EPOCHS + 1):
        generator.train()
        discriminator.train()

        phase = "warmup" if epoch <= WARMUP_EPOCHS else "adversarial"

        total_gen_loss  = 0.0
        total_disc_loss = 0.0

        for imgs in train_loader:
            imgs   = imgs.to(DEVICE)

            with torch.amp.autocast('cuda'):
                ae_out  = autoencoder(imgs)
                gan_out = generator(ae_out)

            # ── Discriminator (adversarial phase only) ─────────────────────────
            if phase == "adversarial":
                opt_d.zero_grad()
                with torch.amp.autocast('cuda'):
                    real_pred = discriminator(imgs)
                    fake_pred = discriminator(gan_out.detach())
                    real_loss = criterion_adv(real_pred, torch.ones_like(real_pred))
                    fake_loss = criterion_adv(fake_pred, torch.zeros_like(fake_pred))
                    disc_loss = (real_loss + fake_loss) * 0.5
                scaler.scale(disc_loss).backward()
                scaler.step(opt_d)
                scaler.update()
                total_disc_loss += disc_loss.item() * imgs.size(0)

            # ── Generator ─────────────────────────────────────────────────────
            opt_g.zero_grad()
            with torch.amp.autocast('cuda'):
                l1_loss   = criterion_l1(gan_out, imgs)
                perc_loss = perceptual(gan_out, imgs)
                if phase == "warmup":
                    gen_loss = l1_loss + 0.1 * perc_loss
                else:
                    fake_pred = discriminator(gan_out)
                    adv_loss  = criterion_adv(fake_pred, torch.ones_like(fake_pred))
                    gen_loss  = l1_loss + 0.1 * perc_loss + 0.001 * adv_loss

            scaler.scale(gen_loss).backward()
            scaler.step(opt_g)
            scaler.update()
            total_gen_loss += gen_loss.item() * imgs.size(0)

        avg_gen_loss  = total_gen_loss  / len(train_dataset)
        avg_disc_loss = total_disc_loss / len(train_dataset) if phase == "adversarial" else 0.0

        # ── Validation ────────────────────────────────────────────────────────
        val_psnr_ae, val_ssim_ae, val_psnr_gan, val_ssim_gan = validate(generator, autoencoder, val_loader)

        line = (f"{epoch},{phase},{avg_gen_loss:.6f},{avg_disc_loss:.6f},"
                f"{val_psnr_ae:.4f},{val_ssim_ae:.4f},"
                f"{val_psnr_gan:.4f},{val_ssim_gan:.4f}")
        log_file.write(line + "\n")
        log_file.flush()

        print(f"Epoch [{epoch:03d}/{TOTAL_EPOCHS}] [{phase}] "
              f"G: {avg_gen_loss:.6f} D: {avg_disc_loss:.6f} | "
              f"Val PSNR AE: {val_psnr_ae:.2f} GAN: {val_psnr_gan:.2f} | "
              f"Val SSIM AE: {val_ssim_ae:.4f} GAN: {val_ssim_gan:.4f}")

        generator.train()

print("GAN training complete.")

# ── Save GAN weights ──────────────────────────────────────────────────────────
torch.save(generator.state_dict(),     os.path.join(WEIGHTS_DIR, "generator.pt"))
torch.save(discriminator.state_dict(), os.path.join(WEIGHTS_DIR, "discriminator.pt"))
print("Saved generator.pt and discriminator.pt")

# ── Save result figure: 5 images (Original | AE output | GAN output) ──────────
random.seed(42)
indices = random.sample(range(len(test_dataset)), 5)

generator.eval()
autoencoder.eval()

fig, axes = plt.subplots(6, 5, figsize=(15, 18))
fig.suptitle("GAN: Original vs AE Reconstruction vs GAN Enhanced", fontsize=14)

with torch.no_grad():
    for col, idx in enumerate(indices):
        img     = test_dataset[idx].unsqueeze(0).to(DEVICE)
        ae_out  = autoencoder(img)
        gan_out = generator(ae_out)  # clean AE output at inference

        orig_np  = img.squeeze().permute(1,2,0).cpu().numpy()
        ae_np    = ae_out.squeeze().permute(1,2,0).cpu().numpy()
        gan_np   = gan_out.squeeze().permute(1,2,0).cpu().numpy()

        psnr_ae  = compute_psnr(orig_np, ae_np)
        ssim_ae  = compute_ssim(orig_np, ae_np)
        psnr_gan = compute_psnr(orig_np, gan_np)
        ssim_gan = compute_ssim(orig_np, gan_np)

        # Row 0: Original label
        axes[0, col].text(0.5, 0.5, "Original", ha='center', va='center', fontsize=11, fontweight='bold')
        axes[0, col].axis('off')

        # Row 1: Original image
        axes[1, col].imshow(orig_np, vmin=0, vmax=1)
        axes[1, col].axis('off')

        # Row 2: AE output label
        axes[2, col].text(0.5, 0.5, "AE Reconstructed", ha='center', va='center', fontsize=11, fontweight='bold')
        axes[2, col].axis('off')

        # Row 3: AE output image
        axes[3, col].imshow(ae_np, vmin=0, vmax=1)
        axes[3, col].axis('off')

        # Row 4: GAN output label
        axes[4, col].text(0.5, 0.5, "GAN Enhanced", ha='center', va='center', fontsize=11, fontweight='bold')
        axes[4, col].axis('off')

        # Row 5: GAN output image + PSNR/SSIM
        axes[5, col].imshow(gan_np, vmin=0, vmax=1)
        axes[5, col].set_xlabel(
            f"AE  PSNR: {psnr_ae:.2f} dB | SSIM: {ssim_ae:.4f}\n"
            f"GAN PSNR: {psnr_gan:.2f} dB | SSIM: {ssim_gan:.4f}",
            fontsize=7
        )
        axes[5, col].set_xticks([])
        axes[5, col].set_yticks([])

plt.tight_layout()
save_path = os.path.join(RESULTS_DIR, "gan_results.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved result figure -> {save_path}")