import torch
from torchvision import datasets, transforms
import numpy as np
from opacus import PrivacyEngine
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

class HeartDiseaseModel(nn.Module):
    def __init__(self,):
        super(HeartDiseaseModel, self).__init__()
        self.fc1 = nn.Linear(11, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x


def pre():
    model = HeartDiseaseModel()

    cleveland = pd.read_csv('/Users/a123/Desktop/coop/COOP-23summer/federated-learning-token/tabnet_pre/standardized_data_v1.csv')
    print('Shape of DataFrame: {}'.format(cleveland.shape))
    print(cleveland.loc[1])

    cleveland.head()

    data = cleveland[~cleveland.isin(['?'])]
    data.loc[280:]
    data = data.dropna(axis=0)

    y = data['target']
    X = data.drop(['target'], axis=1)
    y = y.to_numpy()
    X = X.to_numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

    def df_to_tensor(df):
        return torch.from_numpy(df).float()

    X_traint = df_to_tensor(X_train)
    y_traint = df_to_tensor(y_train)
    X_testt = df_to_tensor(X_test)
    y_testt = df_to_tensor(y_test)

    from torch.utils.data import DataLoader

    train_ds = TensorDataset(X_traint, y_traint)
    test_ds = TensorDataset(X_testt, y_testt)

    # create data loaders
    batch_size = 512
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    privacy_engine = PrivacyEngine()
    loss_fn = nn.BCELoss() # Binary Cross Entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model, optimizer, train_dataloader = privacy_engine.make_private(module=model,optimizer=optimizer,data_loader=train_dataloader,
                                                                noise_multiplier=1.0,   max_grad_norm=1.0, )
    return model, optimizer, train_dataloader, test_dataloader

def get_model():
    model = HeartDiseaseModel()
    return model
# def get_optimizer():
#     return optimizer
# def get_train_dataloader():
#     return train_dataloader
# def get_test_dataloader():
#     return test_dataloader



def train(model, train_dataloader, optimizer):
    loss_fn = nn.BCELoss() # Binary Cross Entropy
    model.train()
    for epoch in range(1, 50):
        losses = []
        predictions = []
        targets = []
        for batch, (data, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = model(data)
            target = target.unsqueeze(1).float() 
            target = target.repeat(1, output.shape[1])
            loss = loss_fn(output, target)
            predicted = torch.round(output)
            predictions.extend(predicted.tolist())
            targets.extend(target.tolist())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        acc = accuracy_score(targets, predictions)
        print('Epoch: {}, Avg. Loss: {:.4f}, Acc: {:.4f}'.format(epoch, np.mean(losses), acc))

def test(model, test_dataloader):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for data, target in test_dataloader:
            output = model(data)
            predicted = torch.round(output)
            predictions.extend(predicted.tolist())
            targets.extend(target.tolist())
    acc = accuracy_score(targets, predictions)
    print('Test Accuracy: {:.4f}'.format(acc))
    print('Confusion Matrix:')
    print(confusion_matrix(targets, predictions))
    print('Classification Report:')
    print(classification_report(targets, predictions))






# train(model, train_dataloader, optimizer)
# test(model, test_dataloader)