import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset yolu
train_dir = "data/NEU-DET/train/images"

# transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# dataset
train_dataset = datasets.ImageFolder(train_dir, transform=transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

# model
model = models.resnet50(pretrained=True)

# son katmanı değiştir
model.fc = nn.Linear(model.fc.in_features, 6)

model = model.to(device)

# loss
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# epoch
epochs = 8

print("Training started...")

for epoch in range(epochs):

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} Loss: {running_loss/len(train_loader)}")


# model kaydet
torch.save(model.state_dict(), "outputs/models/resnet_model.pt")

print("Model saved -> outputs/models/resnet_model.pt")