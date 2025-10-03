import torch
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import build_model
import torch.optim as optim
import torch.nn as nn
import os
from torch.optim.lr_scheduler import StepLR

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
# --- KEY CHANGE 1: Increased epochs for deeper learning ---
EPOCHS = 25
MODEL_SAVE_PATH = "saved_models/fracture_detection_model.pth"

class MuraDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        image = Image.open(img_path).convert("RGB")
        label = int(self.df.iloc[idx]['label'])
        if self.transform:
            image = self.transform(image)
        return image, label

# --- KEY CHANGE 2: Enhanced data augmentation ---
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

if __name__ == '__main__':
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    full_df = pd.read_csv('data/mura_dataset.csv')
    train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42, stratify=full_df.label)
    
    train_dataset = MuraDataset(df=train_df, transform=data_transforms['train'])
    val_dataset = MuraDataset(df=val_df, transform=data_transforms['val'])
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    }

    model = build_model().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # --- KEY CHANGE 3: Added a learning rate scheduler ---
    # This will decrease the learning rate by a factor of 0.1 every 7 epochs
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    
    best_val_accuracy = 0.0
    print("[INFO] Starting model training for higher accuracy...")
    
    for epoch in range(EPOCHS):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for images, labels in dataloaders[phase]:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step() # Step the scheduler after each training epoch

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f"Epoch {epoch+1}/{EPOCHS} | {phase.title()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            if phase == 'val' and epoch_acc > best_val_accuracy:
                best_val_accuracy = epoch_acc
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"[INFO] New best model saved with accuracy: {best_val_accuracy:.4f}")

    print(f"\n[INFO] Training complete! Best validation accuracy: {best_val_accuracy:.4f}")