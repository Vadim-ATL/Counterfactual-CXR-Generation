import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageTk
import numpy as np
import threading

import tkinter as tk
from tkinter import filedialog, ttk, messagebox

# Import your model architectures
from models import SiT_models
from diffusers.models import AutoencoderKL
from anatomy_encoder import AnatomyEncoder

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ANAT_CKPT = r"C:\Users\HCI-4\Desktop\MIMIC_Counterfactual\checkpoints\best_anatomy_encoder.pt"
SIT_FINETUNED_CKPT = r"C:\Users\HCI-4\Desktop\SiTXRay\SiT\inpaint_results\checkpoints\best_checkpoint.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# MODEL DEFINITIONS
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

        for p in self.sit.parameters(): p.requires_grad = True
        self.anat_encoder.eval()
        for p in self.anat_encoder.parameters(): p.requires_grad = False

    def _get_anatomy_tokens(self, clean_img, drop_anat=False):
        B = clean_img.shape[0]
        if drop_anat: return self.uncond_anat.expand(B, -1, -1)
        if clean_img.shape[-1] != 256 or clean_img.shape[-2] != 256:
            clean_img = F.interpolate(clean_img, size=(256, 256), mode="bilinear", align_corners=False)
        
        with torch.no_grad():
            z_spatial, _ = self.anat_encoder(clean_img)        
            z_spatial = F.adaptive_avg_pool2d(z_spatial, (16, 16))
            
        x = z_spatial.flatten(2).transpose(1, 2)               
        return self.anat_proj(x) + self.anat_pos_embed

    def forward(self, x_noisy, t, severity, clean_img, drop_anat=False, drop_class=False):
        anat_tokens = self._get_anatomy_tokens(clean_img, drop_anat=drop_anat)
        img_tokens = self.sit.x_embedder(x_noisy) + self.sit.pos_embed
        x = torch.cat([anat_tokens, img_tokens], dim=1)
        t_embed = self.sit.t_embedder(t)
        
        B = x_noisy.shape[0]
        if drop_class:
            y_null = torch.full((B,), 2, dtype=torch.long, device=x_noisy.device)
            y_embed = self.sit.y_embedder(y_null, False)
        else:
            y_healthy = torch.zeros(B, dtype=torch.long, device=x_noisy.device)
            y_sick = torch.ones(B, dtype=torch.long, device=x_noisy.device)
            emb_healthy = self.sit.y_embedder(y_healthy, False)
            emb_sick = self.sit.y_embedder(y_sick, False)
            sev = severity.view(B, 1)
            y_embed = (1.0 - sev) * emb_healthy + sev * emb_sick

        c = t_embed + y_embed
        for block in self.sit.blocks: x = block(x, c)
        img_out = self.sit.final_layer(x[:, self.num_anat_tokens:], c)
        return self.sit.unpatchify(img_out)[:, :4]

@torch.no_grad()
def euler_sample(model, x_source, severity_target, clean_img_512, device, euler_steps=20, cfg_scale=4.0, edit_strength=0.85):
    v_B = x_source.shape[0]
    initial_noise = torch.randn_like(x_source)
    dt = 1.0 / euler_steps

    t_start = 1.0 - edit_strength
    sample_x = t_start * x_source + (1.0 - t_start) * initial_noise
    start_step = int(euler_steps * t_start)

    for i in range(start_step, euler_steps):
        t_val = i / euler_steps
        vec_t = torch.full((v_B,), t_val, device=device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_cond = model(x_noisy=sample_x, t=vec_t, severity=severity_target, clean_img=clean_img_512, drop_anat=False, drop_class=False)
            pred_uncond = model(x_noisy=sample_x, t=vec_t, severity=severity_target, clean_img=clean_img_512, drop_anat=True, drop_class=True)
        
        v_pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
        sample_x = sample_x + v_pred * dt
        
    return sample_x

# ==============================================================================
# TKINTER APP
# ==============================================================================
class XRayGridSimulatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MIMIC-CXR Counterfactual Timeline Generator")
        self.root.geometry("1450x450")
        self.root.configure(padx=15, pady=15)
        
        self.original_pil_image = None
        self.grid_images = []  
        self.generated_filenames = [] 
        self.tf_resize = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
        
        self.build_ui()
        self.root.after(100, self.load_models)

    def build_ui(self):
        control_frame = ttk.Frame(self.root, width=280)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))

        ttk.Button(control_frame, text="1. Upload Patient X-Ray", command=self.load_image).pack(fill=tk.X, pady=(0, 15))

        ttk.Label(control_frame, text="Simulation Mode:").pack(anchor=tk.W)
        self.mode_var = tk.StringVar(value="Progression (0.0 -> 1.0)")
        self.mode_dropdown = ttk.Combobox(control_frame, textvariable=self.mode_var, state="readonly")
        self.mode_dropdown['values'] = ("Progression (0.0 -> 1.0)", "Regression (1.0 -> 0.0)")
        self.mode_dropdown.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(control_frame, text="Edit Strength:").pack(anchor=tk.W)
        self.edit_var = tk.DoubleVar(value=0.92)
        ttk.Scale(control_frame, from_=0.4, to=1.0, variable=self.edit_var, command=lambda v: self.edit_lbl.config(text=f"{float(v):.2f}")).pack(fill=tk.X)
        self.edit_lbl = ttk.Label(control_frame, text="0.92")
        self.edit_lbl.pack(anchor=tk.E, pady=(0, 15))

        ttk.Label(control_frame, text="CFG Scale:").pack(anchor=tk.W)
        self.cfg_var = tk.DoubleVar(value=9.5)
        ttk.Scale(control_frame, from_=1.0, to=15.0, variable=self.cfg_var, command=lambda v: self.cfg_lbl.config(text=f"{float(v):.1f}")).pack(fill=tk.X)
        self.cfg_lbl = ttk.Label(control_frame, text="9.5")
        self.cfg_lbl.pack(anchor=tk.E, pady=(0, 15))

        ttk.Label(control_frame, text="Euler Steps:").pack(anchor=tk.W)
        self.steps_var = tk.IntVar(value=25)
        ttk.Scale(control_frame, from_=10, to=50, variable=self.steps_var, command=lambda v: self.steps_lbl.config(text=f"{int(float(v))}")).pack(fill=tk.X)
        self.steps_lbl = ttk.Label(control_frame, text="25")
        self.steps_lbl.pack(anchor=tk.E, pady=(0, 20))

        self.generate_btn = ttk.Button(control_frame, text="2. Generate Severity Grid", command=self.start_generation, state=tk.DISABLED)
        self.generate_btn.pack(fill=tk.X, pady=(0, 10))

        self.save_btn = ttk.Button(control_frame, text="3. Save Grid Images", command=self.save_grid_images, state=tk.DISABLED)
        self.save_btn.pack(fill=tk.X)
        
        self.status_lbl = ttk.Label(control_frame, text="Loading models to GPU...", foreground="blue")
        self.status_lbl.pack(pady=10)

        grid_container = ttk.LabelFrame(self.root, text="Disease Timeline Strip (No Labels)")
        grid_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas_label = tk.Label(grid_container, text="Please upload an image and click generate.", bg="gray20", fg="white")
        self.canvas_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def load_models(self):
        try:
            self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE).eval()
            
            self.anat_enc = AnatomyEncoder().to(DEVICE).eval()
            anat_state_raw = torch.load(ANAT_CKPT, map_location=DEVICE)
            anat_state = anat_state_raw.get("anatomy_enc", anat_state_raw)
            self.anat_enc.load_state_dict(anat_state, strict=False)

            sit_base = SiT_models["SiT-XL/2"](num_classes=2).to(DEVICE)
            self.model = SiTAnatomyWrapper(sit_base, self.anat_enc, sit_hidden_dim=1152).to(DEVICE).eval()

            if os.path.exists(SIT_FINETUNED_CKPT):
                ckpt = torch.load(SIT_FINETUNED_CKPT, map_location=DEVICE)
                sit_state = ckpt.get("model_state_dict", ckpt)
                cleaned_sit_state = {k.replace("module.", ""): v for k, v in sit_state.items()}
                self.model.load_state_dict(cleaned_sit_state, strict=False)
            
            self.status_lbl.config(text="Models Ready!", foreground="green")
            if self.original_pil_image is not None:
                self.generate_btn.config(state=tk.NORMAL)
        except Exception as e:
            self.status_lbl.config(text="Error loading models!", foreground="red")
            messagebox.showerror("Model Load Error", str(e))

    def load_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if not filepath: return
        
        self.original_pil_image = Image.open(filepath).convert("RGB")
        self.grid_images = []
        self.save_btn.config(state=tk.DISABLED)
        
        preview = self.original_pil_image.resize((220, 220))
        tk_img = ImageTk.PhotoImage(preview)
        self.canvas_label.config(image=tk_img, text="")
        self.canvas_label.image = tk_img
        
        if self.status_lbl.cget("text") == "Models Ready!":
            self.generate_btn.config(state=tk.NORMAL)

    def start_generation(self):
        if self.original_pil_image is None: return
        self.generate_btn.config(state=tk.DISABLED, text="Processing Grid...")
        self.save_btn.config(state=tk.DISABLED)
        self.status_lbl.config(text="Computing latent vector shifts...", foreground="blue")
        threading.Thread(target=self.process_grid, daemon=True).start()

    def process_grid(self):
        try:
            edit_str = self.edit_var.get()
            cfg = self.cfg_var.get()
            steps = self.steps_var.get()
            chosen_mode = self.mode_var.get()

            if "Progression" in chosen_mode:
                dial_steps = [0.0, 0.25, 0.50, 0.75, 1.0]
                self.generated_filenames = ["0_original.png", "1_sev_0.00.png", "2_sev_0.25.png", "3_sev_0.50.png", "4_sev_0.75.png", "5_sev_1.00.png"]
            else:
                dial_steps = [1.0, 0.75, 0.50, 0.25, 0.0]
                self.generated_filenames = ["0_original.png", "1_sev_1.00.png", "2_sev_0.75.png", "3_sev_0.50.png", "4_sev_0.25.png", "5_sev_0.00.png"]

            local_grid_images = [self.original_pil_image.resize((220, 220))]
            img_t = self.tf_resize(self.original_pil_image).unsqueeze(0).to(DEVICE)
            img_256 = F.interpolate(img_t, (256, 256), mode="bilinear", align_corners=False)
            
            with torch.no_grad():
                v_x1_raw = self.vae.encode(img_256 * 2.0 - 1.0).latent_dist.sample()
                v_x1 = v_x1_raw.mul_(0.18215)

                for step_val in dial_steps:
                    v_sev = torch.tensor([step_val], dtype=torch.float32, device=DEVICE)
                    
                    sample_latent = euler_sample(
                        model=self.model, x_source=v_x1, severity_target=v_sev, clean_img_512=img_t, 
                        device=DEVICE, euler_steps=int(steps), cfg_scale=cfg, edit_strength=edit_str
                    )
                    
                    decoded = torch.clamp((self.vae.decode(sample_latent / 0.18215).sample + 1.0) / 2.0, 0.0, 1.0)
                    out_np = decoded[0].cpu().permute(1, 2, 0).numpy() * 255
                    res_pil = Image.fromarray(out_np.astype(np.uint8)).resize((220, 220))
                    local_grid_images.append(res_pil)

            self.grid_images = local_grid_images

            cell_w, cell_h = 220, 220
            total_w = cell_w * len(self.grid_images)
            master_grid = Image.new('RGB', (total_w, cell_h))
            for idx, img in enumerate(self.grid_images):
                master_grid.paste(img, (idx * cell_w, 0))

            self.root.after(0, lambda: self.finish_generation(master_grid))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda e=e: self.error_generation(e))

    def finish_generation(self, master_grid):
        tk_img = ImageTk.PhotoImage(master_grid)
        self.canvas_label.config(image=tk_img, text="", bg="gray15")
        self.canvas_label.image = tk_img 
        self.generate_btn.config(state=tk.NORMAL, text="2. Generate Severity Grid")
        self.save_btn.config(state=tk.NORMAL) 
        self.status_lbl.config(text="Timeline Rendered Successfully!", foreground="green")

    def save_grid_images(self):
        if not self.grid_images or not self.generated_filenames: return
        target_dir = filedialog.askdirectory(title="Select Folder to Export Array Frames")
        if not target_dir: return
        try:
            for idx, img in enumerate(self.grid_images):
                img.save(os.path.join(target_dir, self.generated_filenames[idx]), "PNG")
            messagebox.showinfo("Export Complete", f"Extracted {len(self.grid_images)} panels to:\n{target_dir}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed: {str(e)}")

    def error_generation(self, e):
        self.generate_btn.config(state=tk.NORMAL, text="2. Generate Severity Grid")
        self.status_lbl.config(text="Generation failed", foreground="red")
        messagebox.showerror("Grid Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = XRayGridSimulatorApp(root)
    root.mainloop()