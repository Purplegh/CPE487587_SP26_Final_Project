import os
import sys
import random
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Run configuration ──────────────────────────────────────────────────────────
# Usage: python autoencoder.py <run_number>
# Example: python autoencoder.py 1   (seed=42)
#          python autoencoder.py 2   (seed=43)
#          python autoencoder.py 3   (seed=44)
RUN        = int(sys.argv[1]) if len(sys.argv) > 1 else 1
SEED       = 41 + RUN  # run1=42, run2=43, run3=44

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print(f"Run {RUN} | Seed {SEED}")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = "/home/zn0004/CPE487587_SP26_Final_Project"
TRAIN_DIR   = f"{BASE_DIR}/data/train"
TEST_DIR    = f"{BASE_DIR}/data/test"
RESULTS_DIR = f"{BASE_DIR}/results/run{RUN}"
WEIGHTS_DIR = f"{BASE_DIR}/weights/run{RUN}"
LOGS_DIR    = f"{BASE_DIR}/logs/run{RUN}"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,    exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
IMAGE_SIZE  = 100
LATENT_DIM  = 128
BATCH_SIZE  = 64
EPOCHS      = 100
LR          = 1e-3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
test_dataset  = MalariaDataset(TEST_DIR,  transform=transform)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")

# ── Model ──────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32,  kernel_size=3, stride=2, padding=1),  # 50x50
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 25x25
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 13x13
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
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # 26x26
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 52x52
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3,  kernel_size=3, stride=2, padding=1, output_padding=1),  # 104x104
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 128, 13, 13)
        x = self.deconv(x)
        x = x[:, :, 2:102, 2:102]  # center crop 104x104 -> 100x100
        return x


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# ── Training ───────────────────────────────────────────────────────────────────
model     = Autoencoder().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.MSELoss()

log_path  = os.path.join(LOGS_DIR, "autoencoder.log")

with open(log_path, "w") as log_file:
    log_file.write("epoch,train_loss\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for imgs in train_loader:
            imgs = imgs.to(DEVICE)
            optimizer.zero_grad()
            recon = model(imgs)
            loss  = criterion(recon, imgs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_dataset)

        log_file.write(f"{epoch},{train_loss:.6f}\n")
        log_file.flush()
        print(f"Epoch [{epoch:03d}/{EPOCHS}]  Train Loss: {train_loss:.6f}")

print("Training complete.")

# ── Save weights ───────────────────────────────────────────────────────────────
torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "autoencoder.pt"))
print("Saved autoencoder.pt")

# ── Export ONNX ───────────────────────────────────────────────────────────────
model.eval()
dummy_img    = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
dummy_latent = model.encoder(dummy_img)

torch.onnx.export(
    model.encoder, dummy_img,
    os.path.join(WEIGHTS_DIR, "encoder.onnx"),
    input_names=["image"], output_names=["latent"],
    dynamic_axes={"image": {0: "batch"}, "latent": {0: "batch"}},
    opset_version=11,
)
print("Saved encoder.onnx")

torch.onnx.export(
    model.decoder, dummy_latent,
    os.path.join(WEIGHTS_DIR, "decoder.onnx"),
    input_names=["latent"], output_names=["image"],
    dynamic_axes={"latent": {0: "batch"}, "image": {0: "batch"}},
    opset_version=11,
)
print("Saved decoder.onnx")

# ── Result figure ──────────────────────────────────────────────────────────────
random.seed(42)
indices = random.sample(range(len(test_dataset)), 5)

model.eval()

# Use gridspec to control row heights: small rows for labels, large for images
fig = plt.figure(figsize=(15, 8))
gs  = gridspec.GridSpec(
    4, 5,
    height_ratios=[0.15, 1, 0.15, 1],
    hspace=0.05, wspace=0.05
)
fig.suptitle("Autoencoder: Original vs Reconstructed", fontsize=13, y=1.01)

with torch.no_grad():
    for col, idx in enumerate(indices):
        img      = test_dataset[idx].unsqueeze(0).to(DEVICE)
        recon    = model(img)

        orig_np  = img.squeeze().permute(1,2,0).cpu().numpy()
        recon_np = recon.squeeze().permute(1,2,0).cpu().numpy()

        mse_val  = ((orig_np - recon_np) ** 2).mean()
        psnr_val = 10 * math.log10(1.0 / mse_val) if mse_val > 0 else 100.0
        ssim_val = ssim(orig_np, recon_np, data_range=1.0, channel_axis=2)

        # Row 0: "Original" label
        ax_label_orig = fig.add_subplot(gs[0, col])
        ax_label_orig.text(0.5, 0.5, "Original", ha='center', va='center',
                           fontsize=10, fontweight='bold')
        ax_label_orig.axis('off')

        # Row 1: Original image
        ax_orig = fig.add_subplot(gs[1, col])
        ax_orig.imshow(orig_np, vmin=0, vmax=1)
        ax_orig.axis('off')

        # Row 2: "Reconstructed" label
        ax_label_recon = fig.add_subplot(gs[2, col])
        ax_label_recon.text(0.5, 0.5, "Reconstructed", ha='center', va='center',
                            fontsize=10, fontweight='bold')
        ax_label_recon.axis('off')

        # Row 3: Reconstructed image + PSNR/SSIM below
        ax_recon = fig.add_subplot(gs[3, col])
        ax_recon.imshow(recon_np, vmin=0, vmax=1)
        ax_recon.set_xlabel(f"PSNR: {psnr_val:.2f} dB\nSSIM: {ssim_val:.4f}", fontsize=8)
        ax_recon.set_xticks([])
        ax_recon.set_yticks([])

save_path = os.path.join(RESULTS_DIR, "autoencoder_results.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved result figure -> {save_path}")