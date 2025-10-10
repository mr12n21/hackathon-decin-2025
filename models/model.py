import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import time
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Používám zařízení: {device}")

data_dir = 'data/'
csv_path = os.path.join(data_dir, 'labels.csv')
df = pd.read_csv(csv_path)

mode_map = {'L-mode': 0, 'H-mode': 1, 'ELM': 2}
df['label'] = df['mode'].map(mode_map)

print("Distribuce tříd:")
print(df['mode'].value_counts())

class_counts = df['label'].value_counts().sort_index()
class_weights = 1. / class_counts
class_weights = class_weights / class_weights.sum() * len(class_weights)
class_weights = torch.tensor(class_weights.values, dtype=torch.float).to(device)

class PlasmaDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, use_alpha=True):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.use_alpha = use_alpha

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')

        if self.use_alpha:
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            _, alpha = cv2.threshold(img_cv, 50, 255, cv2.THRESH_BINARY)  # Prah pro světlé části
            alpha = Image.fromarray(alpha).convert('L')  # Alpha jako PIL Image

            if self.transform:
                alpha = np.array(alpha).astype(np.float32)
                alpha = alpha * np.random.uniform(0.8, 1.2)  # Variace intenzity
                alpha = np.clip(alpha, 0, 255).astype(np.uint8)
                alpha = Image.fromarray(alpha)

            # Stack RGB + Alpha
            image = Image.merge('RGBA', (image.split()[0], image.split()[1], image.split()[2], alpha))

        label = row['label']
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.0], std=[0.229, 0.224, 0.225, 1.0]) if use_alpha else  # Pro RGBA
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Pro RGB
])

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df['discharge']))
train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

img_dir = os.path.join(data_dir, 'img')  # Složka s obrázky
train_dataset = PlasmaDataset(train_df, img_dir, transform=transform, use_alpha=True)
test_dataset = PlasmaDataset(test_df, img_dir, transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]), use_alpha=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Upraveno pro RGBA
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)  # 3 třídy
model = model.to(device)

# Loss a optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Krok 5: Trénink
def train_model(epochs=10):
    model.train()
    losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Graf loss
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    torch.save(model.state_dict(), 'models/saved_model.pth')
    print("Model uložen do models/saved_model.pth")

def evaluate_model():
    model.eval()
    y_true = []
    y_pred = []
    times = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            start = time.time()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            end = time.time()
            times.append((end - start) / inputs.size(0) * 1000)  # ms per snímek
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Přesnost: {acc:.4f}, F1-score: {f1:.4f}")
    print(f"Průměrná rychlost: {np.mean(times):.2f} ms/snímek")
    
    # Confusion matrix graf
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predikováno')
    plt.ylabel('Skutečné')
    plt.show()
    
    # Stabilita: Pro seskupené data (příklad pro jeden discharge)
    sample_discharge = test_df['discharge'].unique()[0]
    seq_df = test_df[test_df['discharge'] == sample_discharge].sort_values('time')
    seq_inputs = [test_dataset.__getitem__(i)[0].unsqueeze(0).to(device) for i in seq_df.index - test_df.index[0]]  # Index adjust
    seq_preds = [model(inp).argmax(1).item() for inp in seq_inputs]
    print(f"Stabilita pro discharge {sample_discharge}: Predikce sekvence: {seq_preds}")
    # Zde můžeš detekovat přechody (diff mezi po sobě jdoucími)

# Spuštění
if __name__ == "__main__":
    train_model(epochs=10)  # Uprav epochs
    evaluate_model()

    # Bonus: Test bez alpha pro porovnání přesnosti
    print("\nTest bez alpha pro porovnání:")
    train_dataset_no_alpha = PlasmaDataset(train_df, img_dir, transform=transform, use_alpha=False)
    test_dataset_no_alpha = PlasmaDataset(test_df, img_dir, transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]), use_alpha=False)
    # Zde retrénuj model bez alpha (uprav conv1 na 3 kanály) a porovnej
    # (Pro jednoduchost: Spusť znovu s use_alpha=False)