import pandas as pd

import numpy as np

from PIL import Image

import torch

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms



class MIMICCounterfactualDataset(Dataset):

    MEAN = [0.5056, 0.5056, 0.5056]  # CXR-specific

    STD  = [0.2523, 0.2523, 0.2523]

   

    def __init__(

        self,

        manifest_path: str,

        split: str = "train",           # "train" | "val"

        condition_filter: str = None,   # None | "Healthy" | "Pneumonia"

        augment: bool = True,

    ):

        self.manifest = pd.read_csv(manifest_path)

        self.split = split

       

        # Filter by split (our_split uses "train" and "val" now)

        df = self.manifest[self.manifest["our_split"] == split].copy()

       

        # Optional filters

        if condition_filter:

            df = df[df["condition"] == condition_filter]

       

        self.df = df.reset_index(drop=True)

        self.augment = augment and (split == "train")

       

        self.transform = self._build_transform()

       

        print(f"[Dataset] split={split} | images={len(self.df)} | augment={self.augment}")

        print(f"  conditions: {self.df['condition'].value_counts().to_dict()}")



    def _build_transform(self):

        ops = []

        # 🟢 REMOVED transforms.Resize() because images are already 512x512 locally!

        if self.augment:

            ops += [

                transforms.RandomHorizontalFlip(p=0.5),

                transforms.RandomAffine(

                    degrees=5,

                    translate=(0.05, 0.05),

                    scale=(0.95, 1.05),

                ),

                transforms.ColorJitter(

                    brightness=0.15,

                    contrast=0.15,

                ),

            ]

        ops += [

            transforms.ToTensor(),

            transforms.Normalize(mean=self.MEAN, std=self.STD),

        ]

        return transforms.Compose(ops)



    def __len__(self):

        return len(self.df)



    def __getitem__(self, idx):

        row = self.df.iloc[idx]

       

        img = Image.open(row["filepath"]).convert("RGB")

        img_tensor = self.transform(img)

       

        # 🟢 Map string condition to binary label dynamically

        label_int = 1 if row["condition"] == "Pneumonia" else 0

       

        return {

            "image":     img_tensor,

            "label":     torch.tensor(label_int, dtype=torch.long),

            "condition": row["condition"],

            "dicom_id":  row["dicom_id"],

            "filepath":  row["filepath"],

        }





def get_dataloaders(

    manifest_path: str,

    batch_size: int = 16, # Increased default

    num_workers: int = 8, # Increased default for SSDs

    condition_filter: str = None,

):

    train_ds = MIMICCounterfactualDataset(

        manifest_path, split="train",

        condition_filter=condition_filter,

        augment=True,

    )

    # 🟢 Point test loader to the "val" split we generated

    test_ds = MIMICCounterfactualDataset(

        manifest_path, split="val",

        condition_filter=condition_filter,

        augment=False,

    )

   

    train_loader = DataLoader(

        train_ds, batch_size=batch_size,

        shuffle=True, num_workers=num_workers,

        pin_memory=True, drop_last=True,

    )

    test_loader = DataLoader(

        test_ds, batch_size=batch_size,

        shuffle=False, num_workers=num_workers,

        pin_memory=True,

    )

    return train_loader, test_loader 

