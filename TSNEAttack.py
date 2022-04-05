import torch
import numpy as np
from torch import nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

def sample_loader(full_dataset, subset_size = 1000, batch_size = 64):
    select_idx = np.random.choice(len(full_dataset), subset_size, replace = False) # subset sample indices generated randomly 
    #print(select_idx)
    # create subset for training
    train_data = Subset(full_dataset,select_idx)
    train_loader = DataLoader(train_data, shuffle = True, batch_size=batch_size)
    return train_loader

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # update stored result
        test_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        #if batch == size - 1:
        #    print(f"loss: {loss:>7f}]")
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    # print result
    test_loss /= num_batches
    correct /= size
    print(f"Training Performance: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Performance: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
def train_model(train_loader, test_loader, model, loss_fn, optimizer, epochs = 10, show_test = False):
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        if show_test:
            test(test_loader, model, loss_fn)
    test(test_loader, model, loss_fn)
       
    print("Done!")
    
def predict_result(dataloader, model, num_class = 6):
    size = len(dataloader.dataset)
    model.eval()
    idx = 0
    res = np.zeros([size,num_class])
    y_true = np.zeros(size)
    with torch.no_grad():
        for X, y in dataloader:
            X= X.to(device)
            #print(X.shape)
            pred = model(X)
            pred = pred.to('cpu').numpy()
            #print(pred.shape)
            res[idx:idx+y.shape[0]] = pred
            y_true[idx:idx+y.shape[0]] = y
            idx = idx + y.shape[0]
    print(res.shape)
    return res, y_true

# create dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.9651, 0.9743, 0.9681],std=[0.1573, 0.1262, 0.1367]), # tsne
     transforms.Grayscale(num_output_channels=3),
     #transforms.RandomHorizontalFlip(0.5),
     #transforms.RandomVerticalFlip(0.5),
     transforms.RandomRotation(180),
     #transforms.Resize((224,224))
    ])

test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.9651, 0.9743, 0.9681],std=[0.1573, 0.1262, 0.1367]), # tsne
     transforms.Grayscale(num_output_channels=3),
     #transforms.Resize((224,224))
    ])
#tsne_train = datasets.ImageFolder('./data/tsne/train/', transform=transform)
#tsne_test = datasets.ImageFolder('./data/tsne/test/', transform=transform)

# Load data
category_type = 'All_Model' # Combine_Model, ResNet, All_Model, Params, ConvDepth, Hyper/BatchSize, Hyper/Optim/64
tsne_train = datasets.ImageFolder('./data/tsne/' + category_type + '/train/', transform=transform)
tsne_test = datasets.ImageFolder('./data/tsne/' + category_type + '/test/', transform=test_transform)
print(tsne_test.classes)

# Create dataloader
batch_size = 32
train_loader = DataLoader(tsne_train, shuffle = True, batch_size=batch_size)
test_loader = DataLoader(tsne_test, batch_size=batch_size)

model = models.resnet18(pretrained = True)
feature_size = 512
num_classes = len(tsne_test.classes)
model.classifier[1] =  nn.Linear(feature_size, num_classes)
model.to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
#optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum = 0.9)
epochs = 20
train_model(train_loader, test_loader, model, loss_fn = loss_fn, optimizer = optimizer, epochs = epochs, show_test = True)

res, y_true = predict_result(test_loader, model,num_classes)
pred_lbl = np.argmax(res,axis = 1)
label_names = tsne_test.classes
print(classification_report(y_true, pred_lbl, target_names=label_names))

cm = confusion_matrix(y_true, pred_lbl, normalize = 'true')
fig, ax = plt.subplots(figsize=(10, 10))
plt.rcParams.update({'font.size': 20})
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(ax=ax, xticks_rotation = 30 )
plt.show()