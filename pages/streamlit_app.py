import streamlit as st
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
from streamlit_drawable_canvas import st_canvas
import cv2
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from model import transform_image, predict_image
import pandas as pd
import plotly.graph_objects as go
import gc

# Front-end
def main():
    st.title("Deep Learning Optimization Algorithms")

    st.sidebar.header("Train Model")
    n_epochs = st.sidebar.slider("Number of Epochs", 1, 50, 20)
    learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)
    optimizer_type = st.sidebar.selectbox("Optimizer", ["SGD", "Adam"])
    stroke_width = st.sidebar.slider("Brush width: ", 10, 30, 20)
    drawing_mode = st.sidebar.checkbox("Drawing mode ?", True)

    if st.sidebar.button("Train"):
        train_model(n_epochs, learning_rate, optimizer_type)

    st.sidebar.header("Test Model")
    test_count = st.sidebar.slider("Number of Test Samples", 1, 20, 10)

    if st.sidebar.button("Test"):
        test_model(test_count)


# Back-end
class CTDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.dataset = datasets.MNIST(root='./MNIST/processed', train=train, transform=self.transform, download=True)

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
    train_loader = DataLoader(CTDataset(train=True), batch_size=64, shuffle=True)
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

#Enable garbage collection
gc.enable()

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet8(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # 1 x 28 x 28
        self.conv1 = conv_block(in_channels, 64) # 64 x 28 x 28
        self.conv2 = conv_block(64, 128, pool=True) # 128 x 14 x 14
        self.res1 = nn.Sequential(conv_block(128, 128), 
                                  conv_block(128, 128)) # 128 x 14 x 14
        
        self.conv3 = conv_block(128, 256, pool=True)  # 256 x 7 x 7
        self.res2 = nn.Sequential(conv_block(256, 256), 
                                  conv_block(256, 256)) # 256 x 7 x 7
        
        self.classifier = nn.Sequential(nn.MaxPool2d(7),  # 256 x 1 x 1 since maxpool with 7x7
                                        nn.Flatten(),    # 256*1*1 
                                        nn.Dropout(0.2),
                                        nn.Linear(256, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out



def transform_image(image):
    stats = ((0.1307), (0.3081))
    my_transforms = T.Compose([

                        T.ToTensor(),
                        T.Normalize(*stats)

                        ])

    return my_transforms(image)




@st.cache
def initiate_model():

    # Initiate model
    in_channels = 1
    num_classes = 10
    model = ResNet8(in_channels, num_classes)
    device = torch.device('cpu')
    PATH = 'mnist-resnet.pth'
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.eval()

    return model




def predict_image(img):
    
    # Convert to a batch of 1
    xb = img.unsqueeze(0)

    model = initiate_model()

    # Get predictions from model
    yb = model(xb)
    # apply softamx
    yb_soft = F.softmax(yb, dim=1)
    # Pick index with highest probability
    confidence , preds  = torch.max(yb_soft, dim=1)
    gc.collect()
    # Retrieve the class label, confidence and probabilities of all classes using sigmoid 
    return preds[0].item(), math.trunc(confidence.item()*100), torch.sigmoid(yb).detach()

if __name__ == "__main__":
    main()
