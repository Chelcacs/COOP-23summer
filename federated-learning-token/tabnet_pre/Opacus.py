#!/usr/bin/env python
# coding: utf-8



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
input_size = 11 # number of features
output_size = 1

model = torch.nn.Sequential(torch.nn.Linear(input_size,64),
                            torch.nn.ReLU(),torch.nn.Linear(64,32),
                            torch.nn.ReLU(),torch.nn.Linear(32,16),
                            torch.nn.ReLU(),torch.nn.Linear(16,16),
                            torch.nn.Sigmoid())



import pandas as pd
cleveland = pd.read_csv('C:/Users/siddh/heart_statlog_cleveland_hungary_final.csv')



print( 'Shape of DataFrame: {}'.format(cleveland.shape))
print (cleveland.loc[1])

cleveland.head()

data = cleveland[~cleveland.isin(['?'])]
data.loc[280:]
data = data.dropna(axis=0)


# In[4]:


y = (data['target'])
X = data.drop(['target'],axis=1)
y=y.to_numpy()
X=X.to_numpy()
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[5]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=5)


# In[6]:


def df_to_tensor(df):
    return torch.from_numpy(df).float()

X_traint = df_to_tensor(X_train)
y_traint = df_to_tensor(y_train)
X_testt = df_to_tensor(X_test)
y_testt = df_to_tensor(y_test)


# In[7]:


X_traint


# In[8]:


from torch.utils.data import DataLoader

train_ds = TensorDataset(X_traint, y_traint)
test_ds = TensorDataset(X_testt, y_testt)

# create data loaders
batch_size = 5
train_dataloader = DataLoader(train_ds, batch_size, shuffle=True)
test_dataloader = DataLoader(test_ds, batch_size, shuffle=False)
# train_dataloader = DataLoader(X_train, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(y_train, batch_size=64, shuffle=True)


# In[9]:


test_dataloader


# In[10]:


privacy_engine = PrivacyEngine()


# In[11]:


loss_fn = nn.BCELoss() # Binary Cross Entropy
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# In[12]:


model, optimizer, dataloader = privacy_engine.make_private(module=model,optimizer=optimizer,data_loader=train_dataloader,
                                                            noise_multiplier=1.0,   max_grad_norm=1.0, )


# In[ ]:


# epochs = 100
# losses = []
# for i in range(epochs):
#     epoch_loss = 0
#     for feat, target in X_train:
#         optim.zero_grad()
#         out = model(feat)
#         loss = loss_fn(out, target.unsqueeze(1))
# #         accuracy =binary_accuracy(feat, target)
#         epoch_loss += loss.item()
#         loss.backward()
#         optim.step()
#     losses.append(epoch_loss)
#     # print loss every 10 
#     if i % 10 == 0:
#         print(f"Epoch: {i}/{epochs}, Loss = {loss:.5f}")


# In[13]:


model


# In[ ]:


# def train(args, model, train_loader, optimizer, privacy_engine, epoch):
#     criterion = nn.CrossEntropyLoss()
#     losses = []
#     accuracies = []
#     device = torch.device(args.device)
#     model = model.train().to(device)

#     for data, label in tqdm(train_loader):
#         data = data.to(device)
#         label = label.to(device)

#         optimizer.zero_grad()
#         predictions = model(data).squeeze(1)
#         loss = criterion(predictions, label)
#         acc = binary_accuracy(predictions, label)

#         loss.backward()
#         optimizer.step()

#         losses.append(loss.item())
#         accuracies.append(acc.item())

#     if not args.disable_dp:
#         epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
#         print(
#             f"Train Epoch: {epoch} \t"
#             f"Train Loss: {np.mean(losses):.6f} "
#             f"Train Accuracy: {np.mean(accuracies):.6f} "
#             f"(ε = {epsilon:.2f}, δ = {args.delta})"
#         )
#     else:
#         print(
#             f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f} ] \t Accuracy: {np.mean(accuracies):.6f}"
#         )


# In[ ]:


mean_accuracy = 0
for epoch in range(1, args.epochs + 1):
    train(args, model, train_loader, optimizer, privacy_engine, epoch)
    mean_accuracy = evaluate(args, model, test_loader)
    

torch.save(mean_accuracy, "run_results_imdb_classification.pt")


# In[23]:


def train(model, train_dataloader, optimizer, epoch, device, delta):
    model.train()
    criterion = torch.nn.BCELoss()
    losses = []
    for idx,(data, target) in enumerate(tqdm(train_dataloader)):
#         data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
#         output = output.squeeze()
        target = target.unsqueeze(1).float() 
#         target = target.view_as(output)
        target = target.repeat(1, output.shape[1])
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    epsilon= float(privacy_engine.get_epsilon(delta)) # optimizer.privacy_engine.get_privacy_spent(delta) 
        
    print(
        f"Train Epoch: {epoch} \t"
        f"Loss: {np.mean(losses):.6f} "
        f"(ε = {epsilon:.2f}, δ = {delta})")
    
for epoch in range(1, 11):
    train(model, train_dataloader, optimizer, epoch, device="cpu", delta=1e-5)


# In[24]:


def evaluate_model(model, test_dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    return accuracy

# Training the model
for epoch in range(1, 11):
    train(model, train_dataloader, optimizer, epoch, device="cpu", delta=1e-5)

# Evaluating the model on the test data
accuracy = evaluate_model(model, test_dataloader)
print(f"Accuracy on test data: {accuracy:.2%}")


# In[ ]:


def train_model(train_dl, model,optimizer,epoch,):
    # define the optimization
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    for epoch in range(100):
        # enumerate mini batches
        for (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()


# In[ ]:


for x,(inputs,targets) in enumerate(tqdm(train_dataloader)):
    print(inputs)


# In[ ]:


tqdm(train_dataloader)


# In[30]:


for data,target in test_dataloader:
    print(data)


# In[ ]:


# model.train()
# criterion = torch.nn.BCELoss()
# losses = []
# for (data, target) in enumerate(tqdm(train_dataloader)):
# #         data, target = data.to(device), target.to(device)
        
#     optimizer.zero_grad()
#     output = model(target)
#     loss = criterion(output, target)
#     loss.backward()
#     optimizer.step()
#     losses.append(loss.item())
    
#     epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)
        
#     print(
#         f"Train Epoch: {epoch} \t"
#         f"Loss: {np.mean(losses):.6f} "
#         f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}")
    
# for epoch in range(1, 11):
#     train(model, train_dataloader, optimizer, epoch, device="cpu", delta=1e-5)


# In[44]:


def accuracy(preds, labels):
    print(type(preds))
    print(type(labels))
    print(preds.shape)
    print(labels.shape)
    return (preds == labels).mean()


# In[51]:


def test_model(model,test_loader,device):
    model.eval()
    criterion= torch.nn.BCELoss()
    losses=[]
    top1acc=[]
    with torch.no_grad():
        for data, target in test_dataloader:
        
#             output = model(data)
#         output = output.squeeze()
            target = target.unsqueeze(1).float() 
#         target = target.view_as(output)
            
            output = model(data)
            target = target.repeat(1, output.shape[1])
            loss=criterion(output,target)
#             preds = np.argmax(output.detach().cpu().numpy(), axis=1)
#             labels = target.detach().cpu().numpy()
#             acc = accuracy(output, target)
            preds = output.detach().cpu().numpy()
            target=target.detach().cpu().numpy()
            acc = accuracy(output, target)
            losses.append(loss.item())
            top1_acc.append(acc)
        top1_avg = np.mean(top1_acc)
        
        print(
            f"\tTest set:"
            f"Loss: {np.mean(losses):.6f} "
            f"Acc: {top1_avg * 100:.6f} "
                                        )
        return np.mean(top1_acc)

# # Training the model
# for epoch in range(1, 11):
#     train(model, train_dataloader, optimizer, epoch, device="cpu", delta=1e-5)

# # Evaluating the model on the test data
# accuracy = test_model(model, test_dataloader,device='cpu')
# print(f"Accuracy on test data: {accuracy:.2%}")


# In[52]:


top1_acc = test_model(model, test_dataloader, device='cpu')


# In[ ]:




