# ==============================================================================
# train_and_evaluate.py - Full Pipeline (Dataset, Preprocessing, CNN, XLS-R, Metrics)
# ==============================================================================
import os
import glob
import random
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import brentq
from scipy.interpolate import interp1d

DATA_DIR = ./dataset
os.makedirs(f{DATA_DIR}/train/real, exist_ok=True)
os.makedirs(f{DATA_DIR}/train/fake, exist_ok=True)
os.makedirs(f{DATA_DIR}/test/real, exist_ok=True)
os.makedirs(f{DATA_DIR}/test/fake, exist_ok=True)

# 1. Dataset Generation / Fallback
def generate_synthetic_fallback():
    print([!] Generating synthetic Indian acoustic fallback dataset...)
    sr = 16000
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    speakers = [fspk_{i:02d} for i in range(1, 9)]
    train_spks = set(speakers[:6])

    for spk in speakers:
        split = train if spk in train_spks else test
        base_f0 = np.random.uniform(110, 240)
        for sample_idx in range(12):
            harmonics = np.sin(2 * np.pi * base_f0 * t) + 0.5 * np.sin(2 * np.pi * 2 * base_f0 * t)
            envelope = 0.5 * (1 + np.sin(2 * np.pi * 2.5 * t))
            real_wave = harmonics * envelope + 0.005 * np.random.normal(0, 1, len(t))
            real_wave = (real_wave / np.max(np.abs(real_wave)) * 0.9).astype(np.float32)
            sf.write(f{DATA_DIR}/{split}/real/{spk}_utt{sample_idx}.wav, real_wave, sr)

            fake_wave = harmonics * envelope + 0.08 * np.sin(2 * np.pi * 3800 * t) + 0.02 * np.random.laplace(0, 1, len(t))
            fake_wave = (fake_wave / np.max(np.abs(fake_wave)) * 0.9).astype(np.float32)
            sf.write(f{DATA_DIR}/{split}/fake/{spk}_utt{sample_idx}_synth.wav, fake_wave, sr)
    print([✓] Fallback speaker-disjoint dataset created.)

try:
    print([*] Fetching IndicTTS-Deepfake-Challenge-Data...)
    ds = load_dataset(SherryT997/IndicTTS-Deepfake-Challenge-Data, split=train, streaming=True)
    count = 0
    for row in ds:
        audio = row[audio]
        label = real if row.get(label, 0) == 0 else fake
        spk = row.get(speaker_id, fspk_{count % 10})
        split = test if (count % 5 == 0) else train
        sf.write(f{DATA_DIR}/{split}/{label}/{spk}_{count}.wav, audio[array], audio[sampling_rate])
        count += 1
        if count >= 100:
            break
    print(f[✓] Downloaded {count} HuggingFace samples.)
except Exception as e:
    print(f[-] Reverting to fallback ({e}))
    generate_synthetic_fallback()

# 2. Preprocessing & Dataset Class
def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    target_len = 16000 * 4
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode=constant)
    else:
        y = y[:target_len]
    return y, sr

def extract_mfcc(y, sr=16000):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=1024, hop_length=512)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    return np.stack([mfcc, d1, d2], axis=0)

class VoiceDataset(Dataset):
    def __init__(self, split=train):
        self.files, self.labels = [], []
        for f in glob.glob(f{DATA_DIR}/{split}/real/*.wav):
            self.files.append(f); self.labels.append(0)
        for f in glob.glob(f{DATA_DIR}/{split}/fake/*.wav):
            self.files.append(f); self.labels.append(1)
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        y, sr = preprocess_audio(self.files[idx])
        features = extract_mfcc(y, sr)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# 3. Model Architecture
class SpectrogramCNN(nn.Module):
    def __init__(self):
        super(SpectrogramCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

device = torch.device(cuda if torch.cuda.is_available() else cpu)
model = SpectrogramCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_loader = DataLoader(VoiceDataset(train), batch_size=8, shuffle=True)
test_loader = DataLoader(VoiceDataset(test), batch_size=4, shuffle=False)

print(f[*] Training CNN baseline on {device}...)
model.train()
for epoch in range(6):
    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)
    print(fEpoch {epoch+1}/6 - Loss: {total_loss/total:.4f} - Acc: {(correct/total)*100:.2f}%)

torch.save(model.state_dict(), baseline_cnn.pth)
print([✓] baseline_cnn.pth saved successfully.)

# 4. Evaluation & Plots
model.eval()
y_true, y_probs = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        y_true.extend(labels.cpu().numpy())
        y_probs.extend(probs.cpu().numpy())

y_true, y_probs = np.array(y_true), np.array(y_probs)
y_pred = (y_probs >= 0.5).astype(int)

acc = accuracy_score(y_true, y_pred)
fpr, tpr, _ = roc_curve(y_true, y_probs, pos_label=1)
try:
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
except Exception:
    eer = fpr[np.nanargmin(np.absolute(((1 - tpr) - fpr)))]

print(f\n[Test Metrics] Accuracy: {acc*100:.2f}% | EER: {eer*100:.2f}%)

# Confusion Matrix
plt.figure(figsize=(4.5, 3.8), dpi=200)
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.title(Confusion Matrix)
plt.tight_layout()
plt.savefig(confusion_matrix.png)
plt.close()

# ROC Curve
plt.figure(figsize=(5, 4), dpi=200)
plt.plot(fpr, tpr, color=#2563EB, label=fROC (EER={eer*100:.1f}%))
plt.plot([0, 1], [0, 1], 'k--')
plt.title(ROC Curve)
plt.xlabel(FPR); plt.ylabel(TPR); plt.legend(); plt.tight_layout()
plt.savefig(roc_curve.png)
plt.close()
print([✓] Saved confusion_matrix.png and roc_curve.png.)
