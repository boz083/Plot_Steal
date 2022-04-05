import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


for model_name in ['ResNet18', 'ResNet34', 'ResNet50', 'MobileV2', 'MobileV3L', 'DenseNet121']: 
    print(model_name)
    save_path = './data/tsne/'
    save_path = os.path.join(save_path, model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(500): # total number of embeddings from shadow/target models
        embed = np.load('./data/embeddings/'+ model_name+'/'+model_name+'_'+str(i)+'.npy')
        lbls = np.load('./data/labels/'+ model_name+'/'+model_name+'_'+str(i)+'.npy')
        for length in [2000]:
            for perp in [30]:
                #length = 2000
                random_idx = np.random.choice(embed.shape[0], length, replace = False) # subset sample indices generated randomly 
                embeddings = embed[random_idx]
                labels = lbls[random_idx]
                #st = time.time()
                TSNE_embedded = TSNE(n_components=2,init='random', perplexity = perp).fit_transform(embeddings)
                #print('Time used: ' + str(time.time() - st))
                # plot and save tsne
                plt.figure(figsize=(3,3))
                plt.scatter(TSNE_embedded[:,0],TSNE_embedded[:,1], s= 0.1, c = labels)
                plt.axis('off')
                tsne_name = model_name + '_' + str(i) + '_'+str(length)+'.png'
                tsne_path = os.path.join(save_path, tsne_name)
                plt.savefig(tsne_path, dpi = 100)
      