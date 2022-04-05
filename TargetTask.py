import torch
import os 
import numpy as np
from torch import nn
import torchvision.models as models
from torchvision import datasets
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"


def sample_loader(full_dataset, subset_size = 1000, batch_size = 64, random_seed = 0):
    np.random.seed(random_seed)
    select_idx = np.random.choice(len(full_dataset), subset_size, replace = False) # subset sample indices generated randomly 
    #print(select_idx)
    # create subset for training
    train_data = Subset(full_dataset,select_idx)
    train_loader = DataLoader(train_data, shuffle = True, batch_size=batch_size)
    return train_loader

def train(dataloader, model, loss_fn, optimizer, device, in_cpu = False, return_loss = False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    quarter_batch_num = int(num_batches/4)
    test_loss, correct = 0, 0
    loss_result = []
    for batch, (X, y) in enumerate(dataloader):
        if in_cpu:
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
        if batch % quarter_batch_num == 0:   #print loss every quarter of total batches
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            loss_result.append(loss)# store loss 
    # print result
    test_loss /= num_batches
    correct /= size
    print(f"Training Performance: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    if return_loss:
        return np.array(loss_result)

def test(dataloader, model, loss_fn, device, in_cpu = False, return_acc = False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            if in_cpu:
                X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Performance: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    if return_acc:
        return correct,test_loss
    
def train_model(train_loader, test_loader, model, loss_fn, optimizer, epochs = 10, device = device):
    loss_train = np.zeros(0)
    loss_test = np.zeros(epochs)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train(train_loader, model, loss_fn, optimizer, device, return_loss = True)
        loss_train = np.concatenate([loss_train,train_loss])
        test_acc, test_loss = test(test_loader, model, loss_fn, device, return_acc = True)
        loss_test[t] = test_loss 
    #test_acc, _ = test(test_loader, model, loss_fn, device, return_acc = True)
    print("Done!")
    
    return loss_train, loss_test, test_acc
    
def preprocess_CIFAR(root_dir = "data"):
    # Load and PreProcess CIFAR10 DATA and load to gpu
    transform = transforms.Compose(
    [transforms.ToTensor(),
     #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # cifar10
     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), # default
     #transforms.Resize((224,224))
    ])

    training_data = datasets.CIFAR10(root=root_dir,train=True,download=True,transform=transform)
    test_data = datasets.CIFAR10(root=root_dir, train=False,download=True,transform=transform)

    # convert to torch tensor and load to GPU
    # convert numpy arrays to pytorch tensors
    train_x = torch.stack([xy[0] for xy in training_data]).to(device)
    #train_y = torch.stack([xy[1] for xy in training_data]).to(device)
    test_x = torch.stack([xy[0] for xy in test_data]).to(device)
    #test_y = torch.stack([xy[1] for xy in test_data]).to(device)
    
    # permute channel dimension to pytorch spec (sample_n, channel, row, col)
    #train_x = torch.tensor(training_data.data, dtype = torch.float).permute(0,3,1,2).to(device)
    train_y = torch.tensor(training_data.targets, dtype = torch.long).to(device)
    #test_x = torch.tensor(test_data.data, dtype = torch.float).permute(0,3,1,2).to(device)
    test_y = torch.tensor(test_data.targets, dtype = torch.long).to(device)

    # Combine original training and testing data
    #X = torch.cat([train_x, test_x])
    #X = X.permute(0,3,1,2) # permute channel dimension to pytorch spec (sample_n, channel, row, col)
    #Y = torch.cat([train_y,test_y])
    training_data = TensorDataset(train_x ,train_y)
    test_data = TensorDataset(test_x ,test_y)
    
    return training_data, test_data

def preprocess_CIFAR100(root_dir = "data", max_id=12):
    # Load and PreProcess CIFAR10 DATA and load to gpu
    transform = transforms.Compose(
    [transforms.ToTensor(),
     #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # cifar10
     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), # default
     #transforms.Resize((224,224))
    ])

    training_data = datasets.CIFAR100(root=root_dir,train=True,download=True,transform=transform)
    test_data = datasets.CIFAR100(root=root_dir, train=False,download=True,transform=transform)

    # convert to torch tensor and load to GPU
    # convert numpy arrays to pytorch tensors
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    min_id = max_id - 10
    for x,y in training_data:
        if min_id <= y < max_id:
            train_x.append(x)
            train_y.append(y - min_id)
    for x,y in test_data:
        if min_id <= y < max_id:
            test_x.append(x)
            test_y.append(y - min_id)
    
    train_x = torch.stack(train_x).to(device)
    test_x = torch.stack(test_x).to(device)
    train_y = torch.tensor(train_y, dtype = torch.long).to(device)
    test_y = torch.tensor(test_y, dtype = torch.long).to(device)
    
    # Combine original training and testing data
    #X = torch.cat([train_x, test_x])
    #X = X.permute(0,3,1,2) # permute channel dimension to pytorch spec (sample_n, channel, row, col)
    #Y = torch.cat([train_y,test_y])
    training_data = TensorDataset(train_x ,train_y)
    test_data = TensorDataset(test_x ,test_y)
    
    return training_data, test_data

def preprocess_Fashion(root_dir = "data"):
    # Load and PreProcess CIFAR10 DATA and load to gpu
    transform = transforms.Compose(
    [transforms.ToTensor(),
     #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # cifar10
     #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), # default
     transforms.Normalize((0.286), (0.353)), # Fashion-MNIST
     transforms.Lambda(lambda x: x.repeat(3, 1, 1) ), # to 3 channel 
     #transforms.Resize((42,42)),
    ])

    training_data = datasets.FashionMNIST(root=root_dir,train=True,download=True,transform=transform)
    test_data = datasets.FashionMNIST(root=root_dir, train=False,download=True,transform=transform)

    # convert to torch tensor and load to GPU
    # convert numpy arrays to pytorch tensors
    train_x = torch.stack([xy[0] for xy in training_data]).to(device)
    #train_y = torch.stack([xy[1] for xy in training_data]).to(device)
    test_x = torch.stack([xy[0] for xy in test_data]).to(device)
    #test_y = torch.stack([xy[1] for xy in test_data]).to(device)
    
    # permute channel dimension to pytorch spec (sample_n, channel, row, col)
    #train_x = torch.tensor(training_data.data, dtype = torch.float).permute(0,3,1,2).to(device)
    train_y = torch.tensor(training_data.targets, dtype = torch.long).to(device)
    #test_x = torch.tensor(test_data.data, dtype = torch.float).permute(0,3,1,2).to(device)
    test_y = torch.tensor(test_data.targets, dtype = torch.long).to(device)

    # Combine original training and testing data
    #X = torch.cat([train_x, test_x])
    #X = X.permute(0,3,1,2) # permute channel dimension to pytorch spec (sample_n, channel, row, col)
    #Y = torch.cat([train_y,test_y])
    training_data = TensorDataset(train_x ,train_y)
    test_data = TensorDataset(test_x ,test_y)
    
    return training_data, test_data

def preprocess_SVHN(root_dir = "data"):
    # Load and PreProcess CIFAR10 DATA and load to gpu
    transform = transforms.Compose(
    [transforms.ToTensor(),
     #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # cifar10
     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), # default
     #transforms.Resize((42,42)),
    ])

    training_data = datasets.SVHN(root=root_dir,split='train',download=True,transform=transform)
    test_data = datasets.SVHN(root=root_dir, split='test',download=True,transform=transform)

    # convert to torch tensor and load to GPU
    # convert numpy arrays to pytorch tensors
    train_x = torch.stack([xy[0] for xy in training_data]).to(device)
    #train_y = torch.stack([xy[1] for xy in training_data]).to(device)
    test_x = torch.stack([xy[0] for xy in test_data]).to(device)
    #test_y = torch.stack([xy[1] for xy in test_data]).to(device)
    
    # permute channel dimension to pytorch spec (sample_n, channel, row, col)
    #train_x = torch.tensor(training_data.data, dtype = torch.float).permute(0,3,1,2).to(device)
    train_y = torch.tensor(training_data.labels, dtype = torch.long).to(device)
    #test_x = torch.tensor(test_data.data, dtype = torch.float).permute(0,3,1,2).to(device)
    test_y = torch.tensor(test_data.labels, dtype = torch.long).to(device)

    # Combine original training and testing data
    #X = torch.cat([train_x, test_x])
    #X = X.permute(0,3,1,2) # permute channel dimension to pytorch spec (sample_n, channel, row, col)
    #Y = torch.cat([train_y,test_y])
    training_data = TensorDataset(train_x ,train_y)
    test_data = TensorDataset(test_x ,test_y)
    
    return training_data, test_data

def select_embed(model_name, TOT_sample, feature_size, selected_size):
    total_embeddings = np.zeros([TOT_sample,selected_size,feature_size])
    for i in range(TOT_sample):
        # load embeddings
        embedding_name = model_name + '_' + str(i) + '.npy'
        embedding_path = './data/embeddings/'
        embeddings = np.load(embedding_path + embedding_name)
        #np.random.seed(0)
        select_idx = np.random.choice(embeddings.shape[0], selected_size, replace = False)
        embeddings = embeddings[select_idx]
        total_embeddings[i] = embeddings
    save_name = model_name + '_embeddings' + '.npy'
    save_path = './data/'
    np.save(save_path + save_name, total_embeddings)

# create dataset
training_data, test_data = preprocess_CIFAR()
#training_data, test_data = preprocess_Fashion()
#training_data, test_data = preprocess_SVHN()
#training_data, test_data = preprocess_CIFAR100()

# Create test dataloader (same for all iterations)
batch_size = 512
test_loader = DataLoader(test_data, batch_size=batch_size)
# Define your output variable that will hold the output
out = None

# iterate through different models
embedding_save_size = 4000
batch_sizes = [16, 64, 128, 512]
optim_name = ['ADAM', 'SGD']
for model_name in ['ResNet18','ResNet34','ResNet50','MobileV2', 'MobileV3L', 'DenseNet121']: #
    print(model_name)
    TOT_sample = 500 # number of models trained (TSNE CREATED)
    target_img_num = 20000 # number of image used for target training for each model 
    num_classes = 10 # 10 for cifar10, fashionMNIST
    # training parameters
    epochs = 8
    
    embedding_path = './data/embeddings/CIFAR10_mix/' + model_name + '/'
    # make path if not exist
    if not os.path.exists(embedding_path):
        os.makedirs(embedding_path)
        
    lbl_path = './data/labels/CIFAR10_mix/' + model_name+'/'
    # make path if not exist
    if not os.path.exists(lbl_path):
        os.makedirs(lbl_path)
        
    post_path = './data/posteriors/CIFAR10_mix/' + model_name+'/'
    # make path if not exist
    if not os.path.exists(post_path):
        os.makedirs(post_path)
    
    model_path = './data/models/CIFAR10_mix/' + model_name+'/'
    # make path if not exist
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    # result store
    loss_train_res = []
    loss_test_res = np.zeros([TOT_sample,epochs])
    acc_res = np.zeros(TOT_sample) # store test acc
    # iterate through different sample 
    for i in range(TOT_sample):
        
        batch_size_idx = np.random.randint(4)
        optim_idx = np.random.randint(2)
        
        batch_size = batch_sizes[batch_size_idx]
        
        save_file_name = model_name + '_BS' + str(batch_size) + '_'+optim_name[optim_idx] + '_' + str(i)
        print(save_file_name)

        # create train_loader for sampled data
        train_loader = sample_loader(training_data, subset_size = target_img_num, batch_size = batch_size, random_seed = i)

        # RESNET 18
        if model_name == 'ResNet18':
            model = models.resnet18(pretrained=True)
            feature_size = 512
            model.fc = nn.Linear(feature_size, num_classes)
            model.to(device)

        # RESNET 34
        elif model_name == 'ResNet34':
            model = models.resnet34(pretrained=True)
            feature_size = 512
            model.fc = nn.Linear(feature_size, num_classes)
            model.to(device)

        # RESNET 50
        elif model_name == 'ResNet50':
            model = models.resnet50(pretrained=True)
            feature_size = 2048
            model.fc = nn.Linear(feature_size, num_classes)
            model.to(device)

        # mobilenet_v2
        elif model_name == 'MobileV2':
            model = models.mobilenet_v2(pretrained = True)
            feature_size = 1280
            model.classifier[1] =  nn.Linear(feature_size, num_classes)
            model.to(device)

        # mobilenet_v3_small
        elif model_name == 'MobileV3L':
            model = models.mobilenet_v3_large(pretrained = True)
            feature_size = 1280
            model.classifier[3] =  nn.Linear(feature_size, num_classes)
            model.to(device)

        # mobilenet_v3_small
        elif model_name == 'MobileV3S':
            model = models.mobilenet_v3_small(pretrained = True)
            feature_size = 1024
            model.classifier[3] =  nn.Linear(feature_size, num_classes)
            model.to(device)

        # dense121
        elif model_name == 'DenseNet121':
            model = models.densenet121(pretrained = True)
            feature_size = 1024
            model.classifier =  nn.Linear(feature_size, num_classes)
            model.to(device)

        else:
            print('Wrong model params')

        # training model
        loss_fn = nn.CrossEntropyLoss()
        if optim_idx == 0:
            LR = 1e-3
            optimizer = optim.Adam(model.parameters(), lr=LR)
        else:
            LR = 1e-3
            optimizer = optim.SGD(model.parameters(), lr=LR, momentum = 0.9)
            
        train_loss, test_loss, test_acc = train_model(train_loader, test_loader, model, loss_fn = loss_fn, optimizer = optimizer, epochs = epochs)
        
        # store results
        loss_train_res.append(train_loss)
        loss_test_res[i] = test_loss
        acc_res[i] = test_acc
        
        # save model
        torch.save(model, model_path + save_file_name)
        
        # save posterior
        model.eval()
        counter = 0
        
        with torch.no_grad():
            for X,_ in test_loader:
                posterior = model(X)
                posterior = posterior.to('cpu').numpy()
                #print(posterior.shape)
                np.save(post_path+ save_file_name, posterior)
                break

        # extract embedding

        # Define a hook function. It sets the global out variable equal to the
        # output of the layer to which this hook is attached to.
        def hook(module, input, output):
            global out
            out = output
            return None

        
        # Your model layer has a register_forward_hook that does the registering for you
        # RESNET 18
        if model_name == 'ResNet18':
            model.avgpool.register_forward_hook(hook)

        # RESNET 34
        elif model_name == 'ResNet34':
            model.avgpool.register_forward_hook(hook)

        # RESNET 50
        elif model_name == 'ResNet50':
            model.avgpool.register_forward_hook(hook)

        # mobilenet_v2
        elif model_name == 'MobileV2':
            model.classifier[0].register_forward_hook(hook)

        # mobilenet_v3_small
        elif model_name == 'MobileV3S':
            model.classifier[2].register_forward_hook(hook)

        # mobilenet_v3_large
        elif model_name == 'MobileV3L':
            model.classifier[2].register_forward_hook(hook)

        # dense121
        elif model_name == 'DenseNet121':
            model.features.norm5.register_forward_hook(hook)

        # Then you just loop through your dataloader to extract the embeddings
        embedding_size = feature_size
        embeddings = np.zeros(shape=(0,embedding_size))
        labels = np.zeros(shape=(0))
        for x,y in iter(test_loader):
            global out
            model(x)
            labels = np.concatenate((labels,y.cpu().numpy().ravel()))
            #print(out.detach().cpu().numpy().shape)
            if model_name in ['MobileV2', 'MobileV3S', 'MobileV3L']:
                new_embedding = out.detach().cpu().numpy()# mobilenetv2 
            else:
                new_embedding = out.detach().cpu().numpy()[:,:,0,0] #resnet18-34-50 dense121 
            embeddings = np.concatenate([embeddings, new_embedding],axis=0)
        
        # random select embeddings
        #np.random.seed(i)
        select_idx = np.random.choice(embeddings.shape[0], embedding_save_size, replace = False) # subset sample indices generated randomly 
        embeddings = embeddings[select_idx]
        labels = labels[select_idx]
        
        # save embeddings
        embedding_name = save_file_name +'.npy'
        np.save(embedding_path + embedding_name, embeddings)
        #total_embeddings = np.concatenate([total_embeddings, embeddings],axis=0)

        # save labels
        lbl_name = save_file_name  + '.npy'
        np.save(lbl_path + lbl_name, labels)
        #total_embeddings = np.concatenate([total_embeddings, embeddings],axis=0)
    loss_train_res = np.asarray(loss_train_res)
    print(loss_train_res.shape, loss_test_res.shape)
    np.save('./data/TrainLoss_CIFAR10_mix_'+model_name+'.npy',loss_train_res)
    np.save('./data/TestLoss_CIFAR10_mix_'+model_name+'.npy',loss_test_res)
    np.save('./data/TestAcc_CIFAR10_mix_'+model_name+'.npy',acc_res)