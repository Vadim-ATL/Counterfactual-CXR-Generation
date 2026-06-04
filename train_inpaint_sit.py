import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image, make_grid

from models import SiT_models
from diffusers.models import AutoencoderKL
from anatomy_encoder import AnatomyEncoder
import torchxrayvision as xrv

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MANIFEST_PATH = r"C:\Users\HCI-4\Desktop\MIMIC_Counterfactual\manifest_vision_severity.csv"
ANAT_CKPT     = r"C:\Users\HCI-4\Desktop\MIMIC_Counterfactual\checkpoints\best_anatomy_encoder_10k.pt"
SIT_BASE_CKPT = r"C:\Users\HCI-4\Desktop\SiTXRay\SiT\results\best_ckpts\Base_003-SiT-XL-2-Linear-velocity-None\checkpoints\latest_checkpoint.pt"

OUT_DIR_VIS  = "inpaint_results/visuals"
OUT_DIR_CKPT = "inpaint_results/checkpoints"
os.makedirs(OUT_DIR_VIS, exist_ok=True)
os.makedirs(OUT_DIR_CKPT, exist_ok=True)

BATCH_SIZE      = 4
NUM_EPOCHS      = 50
LR              = 5e-5
WEIGHT_DECAY    = 1e-5
EULER_STEPS     = 20
CFG_SCALE_MAX   = 6.0          
DROP_CLASS_PROB = 0.15
DROP_ANAT_PROB  = 0.25
VAL_SPLIT       = 0.9
ANAT_ENC_SIZE   = 256          
NUM_WORKERS     = 2            

# ==============================================================================
# 1. LONGITUDINAL PAIRED DATASET
# ==============================================================================
class SiTPairedMIMICDataset(Dataset):
    def __init__(self, manifest_path, split="train", image_size=512):
        self.df = pd.read_csv(manifest_path)
        if "our_split" in self.df.columns:
            self.df = self.df[self.df["our_split"] == split].reset_index(drop=True)
            
        self.patient_groups = self.df.groupby('subject_id')
        self.image_size = image_size
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.LANCZOS),
            transforms.ToTensor(),
        ])
        print(f"[Dataset] Initialized '{split}' split: {len(self.df)} samples.")

    def __len__(self):
        return len(self.df)

    def _load_image(self, row):
        img_path = row["filepath"]
        img_tensor = self.transform(Image.open(img_path).convert("RGB"))
        severity = float(row.get("severity", 0.0))
        condition = str(row["condition"]).lower().strip()
        is_healthy = 1 if condition == "healthy" else 0
        return img_tensor, severity, is_healthy

    def __getitem__(self, idx):
        row_A = self.df.iloc[idx]
        subject_id = row_A['subject_id']
        
        img_A, sev_A, health_A = self._load_image(row_A)
        
        patient_history = self.patient_groups.get_group(subject_id)
        if len(patient_history) > 1 and random.random() > 0.5:
            row_B = patient_history[patient_history['dicom_id'] != row_A['dicom_id']].sample(1).iloc[0]
            img_B, sev_B, health_B = self._load_image(row_B)
        else:
            img_B, sev_B, health_B = img_A, sev_A, health_A

        return {
            "img_A": img_A, "severity_A": torch.tensor(sev_A, dtype=torch.float32), "is_healthy_A": torch.tensor(health_A, dtype=torch.bool),
            "img_B": img_B, "severity_B": torch.tensor(sev_B, dtype=torch.float32), "is_healthy_B": torch.tensor(health_B, dtype=torch.bool),
        }

# ==============================================================================
# 2. NATIVE 4-CHANNEL SPATIAL WRAPPER
# ==============================================================================
class SiTAnatomyWrapper(nn.Module):
    def __init__(self, sit_model, anat_encoder, sit_hidden_dim=1152):
        super().__init__()
        self.sit          = sit_model
        self.anat_encoder = anat_encoder

        self.num_anat_tokens = 256          
        self.anat_proj       = nn.Linear(256, sit_hidden_dim)
        self.anat_pos_embed  = nn.Parameter(torch.randn(1, self.num_anat_tokens, sit_hidden_dim) * 0.02)
        self.uncond_anat     = nn.Parameter(torch.randn(1, self.num_anat_tokens, sit_hidden_dim) * 0.02)

        for p in self.sit.parameters():
            p.requires_grad = True

        self.anat_encoder.eval()
        for p in self.anat_encoder.parameters():
            p.requires_grad = False

    def _get_anatomy_tokens(self, clean_img, drop_anat=False):
        B = clean_img.shape[0]
        if drop_anat:
            return self.uncond_anat.expand(B, -1, -1)

        if clean_img.shape[-1] != ANAT_ENC_SIZE or clean_img.shape[-2] != ANAT_ENC_SIZE:
            clean_img = F.interpolate(clean_img, size=(ANAT_ENC_SIZE, ANAT_ENC_SIZE), mode="bilinear", align_corners=False)

        with torch.no_grad():
            z_spatial, _ = self.anat_encoder(clean_img)        
            z_spatial = F.adaptive_avg_pool2d(z_spatial, (16, 16))

        x = z_spatial.flatten(2).transpose(1, 2)               
        x = self.anat_proj(x)
        return x + self.anat_pos_embed

    def forward(self, x_noisy, t, severity, clean_img, drop_anat=False, drop_class=False):
        anat_tokens = self._get_anatomy_tokens(clean_img, drop_anat=drop_anat)

        # Native 4-channel path
        img_tokens = self.sit.x_embedder(x_noisy) + self.sit.pos_embed

        x = torch.cat([anat_tokens, img_tokens], dim=1)
        t_embed = self.sit.t_embedder(t)
        
        B = x_noisy.shape[0]
        if drop_class:
            y_null = torch.full((B,), 2, dtype=torch.long, device=x_noisy.device)
            y_embed = self.sit.y_embedder(y_null, self.sit.training)
        else:
            y_healthy = torch.zeros(B, dtype=torch.long, device=x_noisy.device)
            y_sick = torch.ones(B, dtype=torch.long, device=x_noisy.device)
            
            emb_healthy = self.sit.y_embedder(y_healthy, self.sit.training)
            emb_sick = self.sit.y_embedder(y_sick, self.sit.training)
            
            sev = severity.view(B, 1)
            y_embed = (1.0 - sev) * emb_healthy + (sev) * emb_sick

        c = t_embed + y_embed

        for block in self.sit.blocks:
            x = block(x, c)

        img_out = self.sit.final_layer(x[:, self.num_anat_tokens:], c)
        output = self.sit.unpatchify(img_out)
        return output[:, :4]

# ==============================================================================
# 3. CLASSIFIER & METRICS
# ==============================================================================
def build_classifier(device):
    clf = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch").to(device)
    clf.eval()
    for p in clf.parameters(): p.requires_grad = False
    return clf, clf.pathologies.index("Pneumonia")

def make_diagnostic_classifier(base_clf, pneumonia_idx):
    def diagnostic_classifier(images):
        if images.shape[1] == 3:
            images = (0.2989 * images[:, 0:1] + 0.5870 * images[:, 1:2] + 0.1140 * images[:, 2:3])
        images = (images * 2048.0) - 1024.0
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        outputs = base_clf(images)                                   
        p_pneu = torch.sigmoid(outputs[:, pneumonia_idx])            
        eps = 1e-6
        logit_pneu = torch.log(p_pneu + eps) - torch.log(1.0 - p_pneu + eps)
        return torch.stack([-logit_pneu, logit_pneu], dim=1)  
    return diagnostic_classifier

def compute_counterfactual_metrics(decoded_images, target_labels, classifier):
    with torch.no_grad():
        logits = classifier(decoded_images)
        predictions = torch.argmax(logits, dim=1)
        int_targets = target_labels.long()
        log_loss = F.cross_entropy(logits, int_targets)
        accuracy = (predictions == int_targets).float().mean()
        probs = torch.softmax(logits, dim=1)
        target_probs = probs[torch.arange(len(int_targets)), int_targets]
    return log_loss.item(), accuracy.item(), target_probs.mean().item()

# ==============================================================================
# 4. EULER SAMPLER (85% NOISE IMG2IMG)
# ==============================================================================
@torch.no_grad()
def euler_sample(model, x_source, severity_target, clean_img_512, device, euler_steps=20, cfg_scale=4.0, edit_strength=0.85):
    v_B = x_source.shape[0]
    initial_noise = torch.randn_like(x_source)
    dt = 1.0 / euler_steps

    # Starts at 85% noise to destroy pneumonia but keep heavy bone frequencies
    t_start = 1.0 - edit_strength
    sample_x = t_start * x_source + (1.0 - t_start) * initial_noise
    start_step = int(euler_steps * t_start)

    for i in range(start_step, euler_steps):
        t_val = i / euler_steps
        vec_t = torch.full((v_B,), t_val, device=device)
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_cond = model(
                x_noisy=sample_x, t=vec_t, severity=severity_target, clean_img=clean_img_512,
                drop_anat=False, drop_class=False
            )
            pred_uncond = model(
                x_noisy=sample_x, t=vec_t, severity=severity_target, clean_img=clean_img_512,
                drop_anat=True, drop_class=True 
            )
            
        v_pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
        sample_x = sample_x + v_pred * dt

    return sample_x

# ==============================================================================
# 5. TRAINING LOOP
# ==============================================================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System] Device: {device}")

    sit_base = SiT_models["SiT-XL/2"](num_classes=2).to(device)
    vae      = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    anat_enc = AnatomyEncoder().to(device)

    if os.path.exists(SIT_BASE_CKPT):
        print(f"[Checkpoint] Loading SiT weights from: {SIT_BASE_CKPT}")
        sit_state   = torch.load(SIT_BASE_CKPT, map_location=device, weights_only=False)
        sit_base.load_state_dict(sit_state.get("model", sit_state), strict=False)

    print(f"[Checkpoint] Loading AnatomyEncoder from: {ANAT_CKPT}")
    anat_state = torch.load(ANAT_CKPT, map_location=device)
    anat_enc.load_state_dict(anat_state.get("anatomy_enc", anat_state))

    model = SiTAnatomyWrapper(sit_base, anat_enc, sit_hidden_dim=1152).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)

    base_clf, pneumonia_idx = build_classifier(device)
    diagnostic_classifier   = make_diagnostic_classifier(base_clf, pneumonia_idx)

    full_dataset = SiTPairedMIMICDataset(MANIFEST_PATH, split="train", image_size=512)
    train_size   = int(VAL_SPLIT * len(full_dataset))
    val_size     = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        cfg_scale = min(1.0 + epoch * ((CFG_SCALE_MAX - 1.0) / 20.0), CFG_SCALE_MAX)

        # --- TRAIN ---
        model.train()
        running_train_loss = 0.0
        train_steps        = 0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            img_A = batch["img_A"].to(device)
            img_B = batch["img_B"].to(device)
            target_severity = batch["severity_B"].to(device)

            img_B_256 = F.interpolate(img_B, (256, 256), mode="bilinear", align_corners=False)
            with torch.no_grad():
                x1 = vae.encode(img_B_256 * 2.0 - 1.0).latent_dist.sample().mul_(0.18215)

            B = x1.shape[0]
            drop_class = np.random.rand() < DROP_CLASS_PROB
            drop_anat  = np.random.rand() < DROP_ANAT_PROB

            x0       = torch.randn_like(x1)
            t        = torch.rand(B, device=device)
            t_expand = t.view(B, 1, 1, 1)
            x_t      = t_expand * x1 + (1.0 - t_expand) * x0
            target_v = x1 - x0

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                # 4-channel feed. latent_A is gone.
                pred_v = model(
                    x_noisy=x_t, t=t, severity=target_severity, clean_img=img_A,
                    drop_anat=drop_anat, drop_class=drop_class
                )
                loss = F.mse_loss(pred_v, target_v)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_train_loss += loss.item()
            train_steps        += 1

            if step % 10 == 0:
                print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}] Step {step:04d} | Train loss: {loss.item():.6f}")

        # --- VALIDATION ---
        model.eval()
        running_cf_log_loss  = 0.0
        running_cf_accuracy  = 0.0
        running_cf_prob      = 0.0
        val_steps            = 0

        print(f"[Val] Epoch {epoch:02d} — Bi-directional Counterfactual Evaluation...")
        MAX_VAL_BATCHES = 10 
        
        with torch.no_grad():
            for i, val_batch in enumerate(val_loader):
                if i >= MAX_VAL_BATCHES: break

                v_img_512 = val_batch["img_A"].to(device)
                v_sev_A   = val_batch["severity_A"].to(device)
                v_y_target = torch.where(v_sev_A < 0.5, torch.ones_like(v_sev_A), torch.zeros_like(v_sev_A)) 

                v_img_256 = F.interpolate(v_img_512, (256, 256), mode="bilinear", align_corners=False)
                v_x1 = vae.encode(v_img_256 * 2.0 - 1.0).latent_dist.sample().mul_(0.18215)
                
                # Edit strength set to 0.85
                sample_latent = euler_sample(
                    model=model, x_source=v_x1, severity_target=v_y_target, clean_img_512=v_img_512, 
                    device=device, euler_steps=EULER_STEPS, cfg_scale=cfg_scale, edit_strength=0.85
                )

                decoded = torch.clamp((vae.decode(sample_latent / 0.18215).sample + 1.0) / 2.0, 0.0, 1.0)

                cf_log_loss, cf_acc, cf_prob = compute_counterfactual_metrics(decoded, v_y_target, diagnostic_classifier)
                running_cf_log_loss += cf_log_loss
                running_cf_accuracy += cf_acc
                running_cf_prob      += cf_prob
                val_steps            += 1

        if val_steps > 0:
            epoch_cf_loss = running_cf_log_loss / val_steps
            print(f"[Val] Target Hit: {running_cf_prob / val_steps * 100:.2f}% | Flip Accuracy: {running_cf_accuracy / val_steps * 100:.2f}%")
        else:
            epoch_cf_loss = float('inf')

        # --- VISUAL SAMPLING (WITH PROGRESSION/REGRESSION GRIDS) ---
        print(f"[Val] Generating Continuous Progression & Regression Grids...")
        with torch.no_grad():
            vis_batch = next(iter(val_loader))
            dial_steps = [0.0, 0.25, 0.50, 0.75, 1.0]
            
            # --- PROGRESSION: HEALTHY TO SICK ---
            healthy_mask = vis_batch["severity_A"].to(device) < 0.5
            if healthy_mask.any():
                # Pick exactly 1 healthy patient to demonstrate continuous progression
                h_idx = torch.where(healthy_mask)[0][0] 
                
                v_img_512 = vis_batch["img_A"].to(device)[h_idx].unsqueeze(0)
                v_img_256 = F.interpolate(v_img_512, (256, 256), mode="bilinear", align_corners=False)
                v_x1 = vae.encode(v_img_256 * 2.0 - 1.0).latent_dist.sample().mul_(0.18215)

                progression_grid = [v_img_256[0]] # First image is original
                
                for step_val in dial_steps:
                    v_cf = torch.full((1,), step_val, device=device, dtype=torch.float32)
                    vis_sample = euler_sample(
                        model=model, x_source=v_x1, severity_target=v_cf, clean_img_512=v_img_512, 
                        device=device, euler_steps=EULER_STEPS, cfg_scale=cfg_scale, edit_strength=0.85
                    )
                    vis_decoded = torch.clamp((vae.decode(vis_sample / 0.18215).sample + 1.0) / 2.0, 0.0, 1.0)
                    progression_grid.append(vis_decoded[0])
                
                # Save as a single long 1x6 row: [Original, 0.0, 0.25, 0.50, 0.75, 1.0]
                save_image(make_grid(torch.stack(progression_grid), nrow=len(dial_steps)+1), f"{OUT_DIR_VIS}/epoch_{epoch:02d}_progression.png")

            # --- REGRESSION: SICK TO HEALTHY ---
            sick_mask = vis_batch["severity_A"].to(device) >= 0.5
            if sick_mask.any():
                # Pick exactly 1 sick patient to demonstrate continuous healing
                s_idx = torch.where(sick_mask)[0][0]
                
                v_img_512 = vis_batch["img_A"].to(device)[s_idx].unsqueeze(0)
                v_img_256 = F.interpolate(v_img_512, (256, 256), mode="bilinear", align_corners=False)
                v_x1 = vae.encode(v_img_256 * 2.0 - 1.0).latent_dist.sample().mul_(0.18215)

                regression_grid = [v_img_256[0]] # First image is original
                
                # Reverse dial steps: 1.0 down to 0.0
                for step_val in reversed(dial_steps):
                    v_cf = torch.full((1,), step_val, device=device, dtype=torch.float32)
                    vis_sample = euler_sample(
                        model=model, x_source=v_x1, severity_target=v_cf, clean_img_512=v_img_512, 
                        device=device, euler_steps=EULER_STEPS, cfg_scale=cfg_scale, edit_strength=0.85
                    )
                    vis_decoded = torch.clamp((vae.decode(vis_sample / 0.18215).sample + 1.0) / 2.0, 0.0, 1.0)
                    regression_grid.append(vis_decoded[0])
                
                # Save as a single long 1x6 row: [Original, 1.0, 0.75, 0.50, 0.25, 0.0]
                save_image(make_grid(torch.stack(regression_grid), nrow=len(dial_steps)+1), f"{OUT_DIR_VIS}/epoch_{epoch:02d}_regression.png")

        # --- CHECKPOINT ---
        epoch_train_loss = running_train_loss / train_steps
        epoch_val_loss = epoch_cf_loss 
        
        if epoch_val_loss < best_val_loss and val_steps > 0:
            best_val_loss = epoch_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss,
            }, f"{OUT_DIR_CKPT}/best_checkpoint.pt")
            print(f"[Checkpoint] New best saved! Val loss: {best_val_loss:.6f}")

        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}] Train loss: {epoch_train_loss:.6f} | Val CF loss: {epoch_val_loss:.6f} | Best: {best_val_loss:.6f}")
        print("-" * 70)

if __name__ == "__main__":
    train()
