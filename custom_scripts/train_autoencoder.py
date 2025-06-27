import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -------------------------------
# 1. Data Loader for Keypoints
# -------------------------------
def load_valid_keypoints_from_json(json_folder, num_keypoints=16):
    """
    Load all fully-valid frames from JSON files in pose_json folder.
    Returns: list of frames (N, num_keypoints, 2)
    """
    all_data = []
    for root, _, files in os.walk(json_folder):
        for file in files:
            if not file.endswith(".json"):
                continue
            path = os.path.join(root, file)
            with open(path, 'r') as f:
                data = json.load(f)

            for frame_no, kpts in data.items():
                frame = []
                valid = True
                for i in range(num_keypoints):
                    pt = kpts.get(str(i))
                    if pt is None or any(np.isnan(v) for v in pt):
                        valid = False
                        break
                    frame.append(pt)
                if valid:
                    all_data.append(frame)

    print(f"✅ Loaded {len(all_data)} valid frames from {json_folder}")
    return np.array(all_data, dtype=np.float32)


# -------------------------------
# 2. Dataset Class
# -------------------------------
class KeypointDataset(Dataset):
    def __init__(self, poses):
        self.poses = poses  # shape: (N, num_keypoints, 2)

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        return torch.tensor(self.poses[idx].flatten(), dtype=torch.float32)


# -------------------------------
# 3. Autoencoder Model
# -------------------------------
class KeypointAutoencoder(nn.Module):
    def __init__(self, input_dim=32):  # 16 keypoints × 2
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)


# -------------------------------
# 4. Training Function
# -------------------------------
def train_autoencoder(model, dataset, epochs=200, lr=1e-3, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"[Epoch {epoch:3d}] Loss: {total_loss / len(dataloader):.6f}")


# -------------------------------
# 5. Save Model
# -------------------------------
def save_model(model, path="models/keypoint_autoencoder.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"✅ Model saved to {path}")


# -------------------------------
# 6. Main Function
# -------------------------------
def main():
    json_folder = "pose_json"  # or "pose_json/Pranav2" etc.
    keypoints = load_valid_keypoints_from_json(json_folder, num_keypoints=16)

    dataset = KeypointDataset(keypoints)
    model = KeypointAutoencoder(input_dim=32)

    train_autoencoder(model, dataset, epochs=150)
    save_model(model)


if __name__ == "__main__":
    main()
