import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

from dataset import get_dataloaders
from anatomy_encoder import AnatomyEncoder, PathologyEncoder
from losses import DisentanglementLoss
from pseudo_masks import images_to_masks

MANIFEST = r"C:\Users\HCI-4\Desktop\MIMIC_Counterfactual\manifest.csv"
SAVE_DIR  = Path(r"C:\Users\HCI-4\Desktop\MIMIC_Counterfactual\checkpoints")
SAVE_DIR.mkdir(exist_ok=True)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS     = 50
BATCH_SIZE = 16  # 🟢 Bumped to 16 for faster 20k processing (drop to 8 if OOM)
LR         = 3e-4
NUM_WORKERS = 8  # 🟢 High workers for SSD saturation


# Small MLP used as the adversarial pathology probe on anatomy latent
class PathoProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, x): return self.net(x)


def train():
    print(f"[System] Starting Anatomy Encoder Training on {DEVICE}...")
    train_loader, test_loader = get_dataloaders(
        MANIFEST, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    anatomy_enc  = AnatomyEncoder().to(DEVICE)
    patho_enc    = PathologyEncoder().to(DEVICE)
    patho_probe  = PathoProbe().to(DEVICE)   # adversarial probe
    criterion    = DisentanglementLoss(lambda_ortho=1.0, lambda_seg=5.0, lambda_cls=0.5)

    # Optimizers
    opt_enc   = AdamW(list(anatomy_enc.parameters()) +
                      list(patho_enc.parameters()), lr=LR, weight_decay=1e-4)
    opt_probe = AdamW(patho_probe.parameters(), lr=LR)
    sched     = CosineAnnealingLR(opt_enc, T_max=EPOCHS)

    best_seg_loss = float("inf")

    for epoch in range(EPOCHS):
        anatomy_enc.train(); patho_enc.train(); patho_probe.train()
        totals = {"seg": 0, "ortho": 0, "adv": 0}

        for batch in train_loader:
            imgs   = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            # Generate pseudo ground-truth masks on the fly
            with torch.no_grad():
                masks = images_to_masks(imgs, DEVICE)

            # Zero out both optimizers at the start of the batch
            opt_enc.zero_grad()
            opt_probe.zero_grad()

            # 🚀 SPEED UP: Compute the forward pass ONCE per batch instead of twice
            z_spatial, seg_logits = anatomy_enc(imgs)

            # --- Step 1: Train probe to classify disease from anatomy ---
            # .detach() isolates the encoder from receiving probe gradients
            z_anat_pooled_detach = z_spatial.detach().mean(dim=(2, 3))  # (B, 256)
            probe_pred = patho_probe(z_anat_pooled_detach)
            probe_loss = nn.CrossEntropyLoss()(probe_pred, labels)
            
            probe_loss.backward()
            opt_probe.step()

            # --- Step 2: Train encoder to fool probe + seg + ortho ---
            # Reuse the original z_spatial and seg_logits to retain gradient tracking
            z_patho = patho_enc(z_spatial)                               # (B, 64)
            
            loss, parts = criterion(
                z_spatial, z_patho, seg_logits,
                masks, labels, patho_probe
            )
            
            loss.backward()
            opt_enc.step()

            for k in totals: totals[k] += parts[k]

        sched.step()
        n = len(train_loader)
        print(f"Epoch {epoch+1:03d}/{EPOCHS} | "
              f"seg={totals['seg']/n:.4f} | "
              f"ortho={totals['ortho']/n:.4f} | "
              f"adv={totals['adv']/n:.4f}")

        # Save best by seg loss (anatomy quality proxy)
        if totals["seg"] / n < best_seg_loss:
            best_seg_loss = totals["seg"] / n
            torch.save({
                "epoch": epoch,
                "anatomy_enc": anatomy_enc.state_dict(),
                "patho_enc":   patho_enc.state_dict(),
            }, SAVE_DIR / "best_anatomy_encoder_10k.pt")
            print(f"  --> Saved new best checkpoint (seg loss: {best_seg_loss:.4f})")


if __name__ == "__main__":
    train()