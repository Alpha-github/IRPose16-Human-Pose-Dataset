import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# --- CONFIGURATION ---
NUM_KEYPOINTS = 16
INPUT_DIM = NUM_KEYPOINTS * 2
MODEL_DIR = "keypoint_validator_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- DEVICE SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Using device: {device}")

# --- MODEL ---
class KeypointAutoencoder(nn.Module):
    def __init__(self, input_dim=INPUT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# --- DATA LOADER ---
def load_training_data(json_folder, num_keypoints=NUM_KEYPOINTS):
    all_frames = []
    for subject in os.listdir(json_folder):
        subject_path = os.path.join(json_folder, subject)
        if not os.path.isdir(subject_path):
            continue
        for fname in os.listdir(subject_path):
            if fname.endswith(".json"):
                with open(os.path.join(subject_path, fname)) as f:
                    data = json.load(f)
                    for frame_no, keypoints in data.items():
                        flat = []
                        for k in range(num_keypoints):
                            pt = keypoints.get(str(k))
                            if pt is None:
                                break
                            flat.extend(pt)
                        if len(flat) == 2 * num_keypoints:
                            all_frames.append(flat)
    return np.array(all_frames, dtype=np.float32)

# --- TRAINING FUNCTION ---
def train_autoencoder(X_train, input_dim=INPUT_DIM, epochs=45, batch_size=64, lr=1e-3, val_split=0.2):
    model = KeypointAutoencoder(input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

    full_dataset = TensorDataset(torch.tensor(X_train))
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model.train()
    for epoch in range(epochs):
        model.train()
        train_loss, train_mae = 0, 0
        for batch in train_loader:
            x = batch[0].to(device)
            out = model(x)
            loss = criterion(out, x)
            mae = mae_metric(out, x)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item()
            train_mae += mae.item()

        model.eval()
        val_loss, val_mae = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                out = model(x)
                loss = criterion(out, x)
                mae = mae_metric(out, x)

                val_loss += loss.item()
                val_mae += mae.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_train_mae = train_mae / len(train_loader)
        avg_val_mae = val_mae / len(val_loader)

        print(f"Epoch {epoch + 1:03d} | "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} | "
              f"Train MAE: {avg_train_mae:.4f}, Val MAE: {avg_val_mae:.4f}")

    return model

# --- MAIN ---
if __name__ == "__main__":
    print("🔍 Loading training data...")
    X = load_training_data("pose_json", num_keypoints=NUM_KEYPOINTS)
    print(f"✅ Loaded {len(X)} valid frames")

    print("🎓 Training autoencoder on", device)
    ae = train_autoencoder(X)
    torch.save(ae.state_dict(), os.path.join(MODEL_DIR, "ae.pth"))

    print("🧠 Fitting PCA model...")
    pca = PCA(n_components=NUM_KEYPOINTS)
    pca.fit(X)
    with open(os.path.join(MODEL_DIR, "pca.pkl"), "wb") as f:
        pickle.dump(pca, f)

    print("🤖 Fitting GMM model...")
    gmm = GaussianMixture(n_components=6, covariance_type='full', random_state=42)
    gmm.fit(X)
    with open(os.path.join(MODEL_DIR, "gmm.pkl"), "wb") as f:
        pickle.dump(gmm, f)

    print("✅ All models trained and saved.")
