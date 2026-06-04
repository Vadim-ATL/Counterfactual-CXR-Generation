# pseudo_masks.py
import torch
import torchvision.transforms.functional as TF
import torchxrayvision as xrv

_seg_model = None

def get_seg_model(device="cuda"):
    global _seg_model
    if _seg_model is None:
        # TorchXRayVision PSPNet predicts 14 specific chest structures
        _seg_model = xrv.baseline_models.chestx_det.PSPNet()
        _seg_model = _seg_model.to(device).eval()
    return _seg_model


def images_to_masks(imgs_tensor, device="cuda"):
    """
    Channel map (Verified via PSPNet Heatmaps):
      4  = Left Lung         | 5  = Right Lung
      8  = Heart             | 11 = Mediastinum
      9  = Aorta             | 12 = Trachea
    
    Output structured classes:
      0 = Background
      1 = Lungs
      2 = Central Silhouette (Heart, Mediastinum, Aorta, Trachea)
    """
    model = get_seg_model(device)

    # Convert back to standard grayscale and re-scale to TorchXRayVision specifications (-1024 to 1024)
    mean = torch.tensor([0.5056, 0.5056, 0.5056], device=device).view(1,3,1,1)
    std  = torch.tensor([0.2523, 0.2523, 0.2523], device=device).view(1,3,1,1)
    gray = (imgs_tensor * std + mean)[:, 0:1]
    gray = (gray - 0.5) * 2048.0
    gray = TF.resize(gray, [512, 512])

    with torch.no_grad():
        out = model(gray.to(device))  # (B, 14, 512, 512)

    masks = torch.zeros(imgs_tensor.shape[0], 512, 512,
                        dtype=torch.long, device=device)

    # 🟢 Class 1: Complete Lung Field
    lung_mask = (out[:, 4] > 0.5) | (out[:, 5] > 0.5)

    # 🟢 Class 2: Complete Central Silhouette (Prevents generative bleeding in the middle)
    central_mask = (
        (out[:, 8] > 0.5) |   # Heart
        (out[:, 11] > 0.5) |  # Mediastinum
        (out[:, 9] > 0.5) |   # Aorta
        (out[:, 12] > 0.5)    # Trachea
    )

    # Paint the map (Central structures overwrite overlapping lung hilum pixels cleanly)
    masks[lung_mask] = 1
    masks[central_mask] = 2  

    return masks