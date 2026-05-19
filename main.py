import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# Load CIFAR-100 dataset
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True)

X_train = np.array(trainset.data)
y_train = np.array(trainset.targets)
X_test = np.array(testset.data)
y_test = np.array(testset.targets)

# Filter to 10 classes
selected_classes = [3, 21, 27, 31, 38, 42, 43, 88, 72, 97]  # animal classes
class_names = ['bear', 'chimpanzee', 'crocodile', 'elephant', 'kangaroo', 'leopard', 'lion', 'tiger', 'wolf', 'seal']

train_mask = np.isin(y_train, selected_classes)
X_train, y_train = X_train[train_mask], y_train[train_mask]

test_mask = np.isin(y_test, selected_classes)
X_test, y_test = X_test[test_mask], y_test[test_mask]

label_map = {orig: new for new, orig in enumerate(selected_classes)}
y_train = np.array([label_map[l] for l in y_train])
y_test = np.array([label_map[l] for l in y_test])

# Normalize pixel values to be between 0 and 1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train).permute(0, 3, 1, 2)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test).permute(0, 3, 1, 2)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# Create DataLoaders
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=16, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=16, shuffle=False)

# Visualize some training images
plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i])
    plt.xlabel(class_names[y_train[i]])
plt.savefig('sample_images.png')

# Build the CNN model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3), padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, (3, 3), padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, (3, 3), padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, (3, 3), padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, (3, 3), padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.25),

        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Setup device, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Training on: {device}")

# Train the model
EPOCHS = 30
train_accs, val_accs, train_losses, val_losses = [], [], [], []

for epoch in range(EPOCHS):
    model.train()
    correct, total, running_loss = 0, 0, 0.0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(1) == y_batch).sum().item()
        total += y_batch.size(0)

    train_accs.append(correct / total)
    train_losses.append(running_loss / len(train_loader))

    model.eval()
    correct, total, running_loss = 0, 0, 0.0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            running_loss += loss.item()
            correct += (outputs.argmax(1) == y_batch).sum().item()
            total += y_batch.size(0)

    val_accs.append(correct / total)
    val_losses.append(running_loss / len(test_loader))

    print(f"Epoch {epoch+1}/{EPOCHS} | Train acc: {train_accs[-1]:.4f} | Val acc: {val_accs[-1]:.4f}")

# Plot training & validation accuracy and loss values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig('results.png')
print("Plot saved to results.png")

# Predict on test images
def plot_predictions(index):
    model.eval()
    img_tensor = X_test_t[index].unsqueeze(0).to(device)
    with torch.no_grad():
        pred_probs = model(img_tensor)
    pred_label = class_names[pred_probs.argmax(1).item()]
    true_label = class_names[y_test[index]]

    plt.imshow(X_test[index])
    plt.title(f"True: {true_label} | Pred: {pred_label}")
    plt.axis('off')
    plt.savefig(f'prediction_{index}.png')

for i in range(5):
    plot_predictions(i)

print(f"\nFinal test accuracy: {val_accs[-1]*100:.2f}%")