import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms

def inference_transforms(image):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    return transform(image).unsqueeze(0)

def augmentation_transform():
    augmentation_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomResizedCrop(28, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    return augmentation_transform


def load_mnist_data():
    original_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=augmentation_transform())

    augmented_datasets = []
    num_augmented_samples = 100000 - len(original_dataset) 

    while len(augmented_datasets) * len(original_dataset) < num_augmented_samples:
        augmented_datasets.append(original_dataset)

    full_dataset = ConcatDataset([original_dataset] + augmented_datasets)
    print(f"Total samples in the augmented dataset: {len(full_dataset)}")

    batch_size = 64
    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    assert len(full_dataset) % batch_size == 0, "Dataset size is not divisible by batch_size"

    return train_loader


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def train_model(model, train_loader, criterion, optimizer, epochs=1):
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')


def run_train(train_loader):
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer, epochs=1)

    torch.save(model.state_dict(), 'mnist_cnn.pth')


def main():
    train_loader = load_mnist_data()
    run_train(train_loader)


if __name__ == "__main__":
    main()