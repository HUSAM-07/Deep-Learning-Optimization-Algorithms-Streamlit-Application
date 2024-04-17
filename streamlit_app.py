import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import math
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Front-end
def main():
    page = st.sidebar.radio("Navigation", ["Digit Recognizer", "Train Model"])
    
    if page == "Digit Recognizer":
        digit_recognizer_page()
    elif page == "Train Model":
        train_model_page()

def digit_recognizer_page():
    st.title('Digit Recognizer')
    st.write("This is a simple image classification web app to **recognize the digit** drawn in the canvas.")
    st.markdown('### Draw a digit !')

    st.sidebar.header("Configuration")
    stroke_width = st.sidebar.slider("Brush width: ", 10, 30, 20)
    drawing_mode = st.sidebar.checkbox("Drawing mode ?", True)

    # To plot classes and probabilities
    def plot_fig(df):
        fig = go.Figure(go.Bar(
            x=df[0].tolist(),
            y=list(df.index),
            orientation='h'))

        fig.update_yaxes(type='category')

        # Label axes and title, center the title, change fonts 
        fig.update_layout(    
            title={
                    'text': "Classes vs Probabilities",
                    'y':0.9,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
            xaxis_title="Probabilities",
            yaxis_title="Classes",
            font=dict(
                family="Courier New, monospace",
                size=18,
                # color="RebeccaPurple"
                ))

        st.plotly_chart(fig)

        del fig

    SIZE = 256
    canvas_result = st_canvas(
        fill_color='#000000',
        stroke_width=stroke_width,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=SIZE,
        height=SIZE,
        drawing_mode="freedraw" if drawing_mode else "transform",
        key='canvas')

    if canvas_result.image_data is not None:

        # cv2 METHOD
        # nd_array -> resized nd_array -> grayscale nd_array -> pytorch tensor

        # Resize the image to 28x28 for the model input
        img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
        # Rescaling the image just to view the model input clearly 
        rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) 
        st.write('`Model input (rescaled)`')
        st.image(rescaled)

        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Convert to Pytorch Tensor
        tensor_img = transform_image(img_gray)

        clicked = st.button('Run')

        if clicked:

            with st.spinner(text='In progress'):
                image_transformed = transform_image(img_gray)
                class_label , confidence, probs = predict_image(image_transformed)

            st.info('Run Successful !')

            if confidence > 50:
                st.write('### `Prediction` : ', str(class_label))
                st.write('### `Confidence` : {}%'.format(confidence)) 
                st.write('&nbsp') # new line
                st.write('**Probabilities**')
                df = pd.DataFrame(probs.numpy())
                st.dataframe(df)
                df = df.transpose()
                plot_fig(df)

                del img_gray, image_transformed , class_label, confidence, df 

            else:
                st.write('### `Prediction` : Unable to predict')
                st.write('### `Confidence` : {}%'.format(confidence)) 
                st.write('&nbsp') # new line
                st.write('**Probabilities**')
                df = pd.DataFrame(probs.numpy())
                st.dataframe(df)
                df = df.transpose()
                plot_fig(df)
                del img_gray, image_transformed , class_label, confidence, df

def train_model_page():
    st.title("Deep Learning Optimization Algorithms")

    st.sidebar.header("Train Model")
    n_epochs = st.sidebar.slider("Number of Epochs", 1, 50, 20)
    learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)
    optimizer_type = st.sidebar.selectbox("Optimizer", ["SGD", "Adam"])

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

    predicted


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
    # apply softmax
    yb_soft = F.softmax(yb, dim=1)
    # Pick index with highest probability
    confidence , preds  = torch.max(yb_soft, dim=1)
    # Retrieve the class label, confidence, and probabilities of all classes using sigmoid 
    return preds[0].item(), math.trunc(confidence.item()*100), torch.sigmoid(yb).detach()

if __name__ == "__main__":
    main()