
import streamlit as st
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score

# Front-end
def main():
    st.title("Deep Learning Optimization Algorithms")

    with st.form("Train Model"):
        n_epochs = st.slider("Number of Epochs", 1, 50, 20)
        learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01)
        optimizer_type = st.selectbox("Optimizer", ["SGD", "Adam"])

        if st.button("Train"):
            train_model(n_epochs, learning_rate, optimizer_type)

        st.header("Test Model")
        test_count = st.slider("Number of Test Samples", 1, 20, 10)

        if st.button("Test"):
            test_model(test_count)


# Back-end
class CTDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, num_samples=1000):  # Add num_samples parameter
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.dataset = datasets.MNIST(root='./MNIST/processed', train=train, transform=self.transform, download=True)

        # Keep only num_samples for training
        if train:
            self.dataset = torch.utils.data.Subset(self.dataset, range(num_samples))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]



class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_model(n_epochs, learning_rate, optimizer_type):
    train_loader = DataLoader(CTDataset(train=True, num_samples=1000), batch_size=64, shuffle=True)
    model = MyNeuralNet()
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_type == "SGD":
        optimizer = SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == "Adam":
        optimizer = Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Invalid optimizer type")

    accuracy_list = []
    for epoch in range(n_epochs):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        accuracy = test_model_accuracy()
        accuracy_list.append(accuracy)
        st.write(f"Epoch {epoch+1}/{n_epochs}, Accuracy: {accuracy:.4f}")

    plt.plot(range(1, n_epochs+1), accuracy_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epoch')
    st.pyplot()


def test_model_accuracy():
    test_loader = DataLoader(CTDataset(train=False), batch_size=1, shuffle=True)
    model = MyNeuralNet()
    model.eval()

    predicted_list = []
    target_list = []
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output, 1)
        predicted_list.append(predicted.item())
        target_list.append(target.item())

    accuracy = accuracy_score(target_list, predicted_list)
    return accuracy


def test_model(test_count):
    test_loader = DataLoader(CTDataset(train=False), batch_size=1, shuffle=True)
    model = MyNeuralNet()
    model.eval()

    cols = 4
    rows = int(np.ceil(test_count / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(10, 5))
    for i, (data, target) in enumerate(test_loader):
        if i == test_count:
            break
        output = model(data)
        _, predicted = torch.max(output, 1)
        image = data.squeeze().numpy()
        ax[i // cols, i % cols].imshow(image, cmap='gray')
        ax[i // cols, i % cols].set_title(f'Predicted: {predicted.item()}, Actual: {target.item()}')
        ax[i // cols, i % cols].axis('off')

    plt.tight_layout()
    st.pyplot(fig)


if __name__ == "__main__":
    main()
