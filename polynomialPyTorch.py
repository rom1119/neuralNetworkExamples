import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import time
from scipy.optimize import curve_fit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 1),

        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    


plt.ion()
fig, axs = plt.subplots(1,sharex=True)
ax = axs


model = NeuralNetwork()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
dtype = torch.float

# print(model)





# X_scaler = MinMaxScaler()
# X_scaled = X_scaler.fit_transform(predictedX)
# y_scaler = MinMaxScaler()
# y_scaled = y_scaler.fit_transform(predictedY)

batch_size = 1

class Data(Dataset):
  def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
    # need to convert float64 to float32 else
    # will get the following error
    # RuntimeError: expected scalar type Double but found Float
    self.X = torch.from_numpy(X.astype(np.float32))
    self.y = torch.from_numpy(y.astype(np.float32))
    self.len = self.X.shape[0]
    print('len', self.len)
  def __getitem__(self, index: int) -> tuple:
    return self.X[index], self.y[index]
  def __len__(self) -> int:
    return self.len
  

start = -12.0
end = 12
# start = -100
# end = 460
# start = 16
# end = 365

# X =  np.linspace(16, 365, 20)
X = np.linspace(start, end, 20)
# Y = (np.sin(1*np.pi*X/153) + np.cos(1*np.pi*X/127)) *100
Y = ( np.cos(X))
# Y = Y * 10
# Y = -((1/-6.2)*X**3 + (1/1.)*X**4 - (1/0.1)*X**2 + 1.4)

predictedX = X.reshape(-1, 1)
predictedY = Y.reshape(-1, 1)


data = Data(predictedX, predictedY)
# x = torch.rand(5, 3)
# print(x)
print(predictedX)
print(predictedY)
# plt.scatter(predictedX, predictedY)
train_dataloader = DataLoader(data, batch_size=batch_size)
# test_dataloader = DataLoader(test_data, batch_size=batch_size)
model.train()

def predict(model, X):
    tensor = torch.from_numpy(np.array([[X]]).astype(np.float32))
    # print('model T', tensor)

    pred = model(tensor)
    # print('model T', pred)
    return pred.item()

epochs = 5000
for epoch in range(epochs):
    batch = 0
    size = len(predictedX)
    for i, data in enumerate(train_dataloader):
        # print('data', data)

        # Every data instance is an input + label pair
        inputs, labels = data
        # print('labels', labels)
        # print('labels', labels[0])
        # print('inputs', inputs)
        # print('inputs', inputs[0])

        # Compute prediction error
        pred = model(inputs[0])
        loss = loss_fn(pred, labels[0])

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
          # display statistics

        
    if epoch % 20 == 0:

        # predX = np.linspace(16, 365, (365-16)*4)
        predX = np.linspace(start, end, 150)
        predY = np.array([ predict(model,pX) for pX in predX])

        ax.clear()
        ax.plot(predX, predY)
        ax.plot(X , Y)

        plt.pause(0.001)
    if epoch % 200 == 0:
        print(f'Epochs:{epoch + 1:5d} | ' \
            f'Batches per epoch: {i + 1:3d} | ' \
            f'Loss: {loss / (i + 1):.10f}')
    # if not ((epoch + 1) % (epochs // 5)):





