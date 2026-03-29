import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from prettytable import PrettyTable
import arrow

from transformer import TumorClassifierViT

# -------------------- SETUP --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint_path = 'checkpoint.pth'
best_model_path = 'best_model.pth'

tableData = PrettyTable([
    'Epoch',
    'Time Elapsed (HH:mm:ss)',
    'Training Loss',
    'Training Accuracy',
    'Validation Loss',
    'Validation Accuracy'
])

# -------------------- DATA --------------------
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = ImageFolder('./data/Training', transform=data_transforms)
val_dataset = ImageFolder('./data/Testing', transform=data_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# -------------------- MODEL --------------------
model = TumorClassifierViT(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# -------------------- RESUME LOGIC --------------------
start_epoch = 0
best_val_accuracy = 0.0

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_accuracy = checkpoint['best_val_accuracy']
    print(f"✅ Resuming training from epoch {start_epoch}")

# -------------------- HISTORY --------------------
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

num_epochs = 100
startTime = arrow.now().timestamp()

for epoch in range(start_epoch, num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    val_loss /= len(val_loader)
    val_accuracy = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    elapsed_time = arrow.get(
        arrow.now().timestamp() - startTime
    ).format("HH:mm:ss")

    print(
        f'Epoch [{epoch+1}/{num_epochs}] | '
        f'Time: {elapsed_time} | '
        f'Train Loss: {train_loss:.4f} | '
        f'Train Acc: {train_accuracy:.2%} | '
        f'Val Loss: {val_loss:.4f} | '
        f'Val Acc: {val_accuracy:.2%}'
    )

    if (epoch + 1) % 5 == 0 or epoch == 0:
        tableData.add_row([
            epoch + 1,
            elapsed_time,
            f'{train_loss:.4f}',
            f'{train_accuracy:.4f}',
            f'{val_loss:.4f}',
            f'{val_accuracy:.2%}'
        ])

    # -------------------- SAVE BEST MODEL --------------------
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), best_model_path)

    # -------------------- SAVE CHECKPOINT --------------------
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_accuracy': best_val_accuracy
    }, checkpoint_path)

# -------------------- RESULTS --------------------
print("\nTraining Complete")
print(tableData)

# -------------------- PLOTS --------------------
plt.figure(figsize=(10, 7))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.savefig('loss.png')
plt.close()

plt.figure(figsize=(10, 7))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.legend()
plt.savefig('accuracy.png')
plt.close()

conf_matrix = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(10, 7))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=val_dataset.classes,
    yticklabels=val_dataset.classes
)
plt.savefig('confusion_matrix.png')
plt.close()