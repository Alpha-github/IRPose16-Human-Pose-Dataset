import numpy as np
import pickle
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

class KeypointAutoencoder(nn.Module):
    def __init__(self, input_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 16), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def load_training_data(json_folder, num_keypoints=16):
    all_frames = []

    import os, json
    for fname in os.listdir(json_folder):
        if fname.endswith(".json"):
            with open(os.path.join(json_folder, fname)) as f:
                data = json.load(f)
                for frame_no, keypoints in data.items():
                    flat = []
                    for k in range(num_keypoints):
                        pt = keypoints[str(k)]
                        if pt is None:
                            break
                        flat.extend(pt)
                    if len(flat) == 2 * num_keypoints:
                        all_frames.append(flat)

    return np.array(all_frames, dtype=np.float32)

def train_autoencoder(X_train, input_dim=32, epochs=50):
    model = KeypointAutoencoder(input_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.tensor(X_train))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for batch in loader:
            x = batch[0]
            out = model(x)
            loss = criterion(out, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")
    
    return model

if __name__ == "__main__":
    print("🔍 Loading training data...")
    X = load_training_data("pose_json/Pranav2")
    print(f"✅ Loaded {len(X)} valid frames")

    print("🎓 Training autoencoder...")
    ae = train_autoencoder(X)
    torch.save(ae.state_dict(), "keypoint_validator_models/ae.pth")

    print("🧠 Fitting PCA model...")
    pca = PCA(n_components=16)
    pca.fit(X)
    with open("keypoint_validator_models/pca.pkl", "wb") as f:
        pickle.dump(pca, f)

    print("🤖 Fitting GMM model...")
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42)
    gmm.fit(X)
    with open("keypoint_validator_models/gmm.pkl", "wb") as f:
        pickle.dump(gmm, f)

    print("✅ All models trained and saved.")
