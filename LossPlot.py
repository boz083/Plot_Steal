import os
import numpy as np
import matplotlib.pyplot as plt

# sliding smoothing
def slide_smooth(train_loss, window = 2):
    new_loss = np.zeros(train_loss.shape)
    for t in range(new_loss.shape[1]):
        window_sum = train_loss[:,t:t+window]
        #print(window_sum.shape[1])
        new_loss[:,t] = np.sum(window_sum, axis = 1)/window_sum.shape[1]
        #break
    return new_loss
# tensorboard smoothing
def tensor_smooth(train_loss, weight = 0.2):
    new_loss = np.zeros(train_loss.shape)
    new_loss[:,0] = train_loss[:,0]
    for t in range(1,new_loss.shape[1]):
        new_loss[:,t] = train_loss[:,t-1] * weight + train_loss[:,t] * (1-weight)
    return new_loss
# gaussian noise
def gaussian_smooth(train_loss, std_ratio = 1):
    new_loss = np.zeros(train_loss.shape)
    for t in range(new_loss.shape[1]):
        current_std = np.std(train_loss[:,t])
        #print(current_std)
        layer_noise = np.random.normal(0, std_ratio * current_std, train_loss.shape[0])
        #print(window_sum.shape[1])
        new_loss[:,t] = train_loss[:,t] + layer_noise
        #break
    return new_loss

# original plots
TOT_SAMPLE = 1000
split_idx = int(TOT_SAMPLE/4*3)
dataset = 'CIFAR'
train_path = './data/lossplot/' + dataset + '/train/'
test_path = './data/lossplot/' + dataset + '/test/'
#train_path = './data/lossplot/' + dataset + '_waxis/train/'
#test_path = './data/lossplot/' + dataset + '_waxis/test/'
inference_targets = ['ResNet18', 'ResNet34', 'ResNet50', 'MobileV2', 'MobileV3L', 'DenseNet121']
plt_ymin = 0
plt_ymax = 3.2
epoch_limit = 25
test_limit = int(epoch_limit/5)
for model_name in inference_targets:
    print(model_name)
    train_save = train_path + model_name + '/'
    test_save = test_path + model_name + '/'
    if not os.path.exists(train_save):
        os.makedirs(train_save)
    if not os.path.exists(test_save):
        os.makedirs(test_save)
    
    train_loss = np.load('./data/loss/TrainLoss_'+dataset+'_'+model_name+'.npy')[:,:epoch_limit]
    test_loss = np.load('./data/loss/TestLoss_'+dataset+'_'+model_name+'.npy')[:,:test_limit]
    
    # train
    for idx in range(split_idx):
        ytrain = train_loss[idx]
        xtrain = np.arange(ytrain.shape[0])
        ytest = test_loss[idx]
        xtest = np.arange(1,ytest.shape[0]+1) * (ytrain.shape[0]/ytest.shape[0]) - 1
        plt.figure(figsize=(3,3))
        #plt.ylim([plt_ymin, plt_ymax])
        plt.yticks(np.arange(plt_ymin, plt_ymax, 0.5))
        plt.xlim([-1, epoch_limit + 1])
        plt.xticks(np.arange(0, epoch_limit + 1, 5))
        plt.plot(xtrain,ytrain)
        plt.plot(xtest,ytest)
        plt.axis('off')
        plt.savefig(train_save + model_name + '_' +str(idx)+'.png', dpi = 50)
        plt.close()
        #break
        
    # test
    for idx in range(split_idx, TOT_SAMPLE):
        ytrain = train_loss[idx]
        xtrain = np.arange(ytrain.shape[0])
        ytest = test_loss[idx]
        xtest = np.arange(1,ytest.shape[0]+1) * (ytrain.shape[0]/ytest.shape[0]) - 1
        plt.figure(figsize=(3,3))
        #plt.ylim([plt_ymin, plt_ymax])
        plt.yticks(np.arange(plt_ymin, plt_ymax, 0.5))
        plt.xlim([-1, epoch_limit + 1])
        plt.xticks(np.arange(0, epoch_limit + 1, 5))
        plt.plot(xtrain,ytrain)
        plt.plot(xtest,ytest)
        plt.axis('off')
        plt.savefig(test_save + model_name + '_'+ str(idx)+'_.png', dpi = 50)
        plt.close()
        #break

# defense plots
TOT_SAMPLE = 500
split_idx = int(TOT_SAMPLE/4*3)
dataset = 'SVHN'
method = 'slide' # 
defense_methods = {'slide': slide_smooth, 'tensor': tensor_smooth, 'gaussian': gaussian_smooth}
#train_path = './data/lossplot/defense/' + dataset + '/train/'
test_path = './data/lossplot/defense/' + dataset + '_waxis/' + method + '/'
#train_path = './data/lossplot/' + dataset + '_waxis/train/'
#test_path = './data/lossplot/' + dataset + '_waxis/test/'
plt_ymin = 0
plt_ymax = 3.2
epoch_limit = 25
test_limit = int(epoch_limit/5)
norms = []
for model_name in ['ResNet18', 'ResNet34', 'ResNet50', 'MobileV2', 'MobileV3L', 'DenseNet121']:
    print(model_name)
    #train_save = train_path + model_name + '/'
    test_save = test_path + model_name + '/'
    #if not os.path.exists(train_save):
    #    os.makedirs(train_save)
    if not os.path.exists(test_save):
        os.makedirs(test_save)
    
    train_loss = np.load('./data/loss/TrainLoss_'+dataset+'_'+model_name+'.npy')[:,:epoch_limit]
    test_loss = np.load('./data/loss/TestLoss_'+dataset+'_'+model_name+'.npy')[:,:test_limit]
    
    new_loss = defense_methods[method](train_loss)
    #new_loss = slide_smooth(train_loss,2)
    #new_loss = tensor_smooth(train_loss,0.2)
    #new_loss = gaussian_smooth(train_loss,1)
    l2_norm = np.linalg.norm(train_loss - new_loss, axis = 1)
    print(np.mean(l2_norm))
    norms.append(np.mean(l2_norm))
    train_loss = new_loss
    # test
    for idx in range(split_idx, TOT_SAMPLE):
        ytrain = train_loss[idx]
        xtrain = np.arange(ytrain.shape[0])
        ytest = test_loss[idx]
        xtest = np.arange(1,ytest.shape[0]+1) * (ytrain.shape[0]/ytest.shape[0]) - 1
        plt.figure(figsize=(3,3))
        #plt.ylim([plt_ymin, plt_ymax])
        plt.yticks(np.arange(plt_ymin, plt_ymax, 0.5))
        plt.xlim([-1, epoch_limit + 1])
        plt.xticks(np.arange(0, epoch_limit + 1, 5))
        plt.plot(xtrain,ytrain)
        plt.plot(xtest,ytest)
        #plt.axis('off')
        plt.savefig(test_save + model_name + '_'+ str(idx)+'_.png', dpi = 50)
        plt.close()
        #break
np.save(dataset + '_' + method, np.asarray(norms))
print('AVG L2: ', np.mean(norms))              