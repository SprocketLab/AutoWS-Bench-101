# based on https://github.com/christiansafka/img2vec/blob/master/img2vec_pytorch/img_to_vec.py
from tkinter import Variable
import torch
import numpy as np
from fwrench.embeddings.base_embedding import BaseEmbedding
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sys import getsizeof

class PretrainedModelEmbedding(BaseEmbedding):
    def __init__(self, model_name='resnet-18'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name

        # TODO: change for different models
        self.model = models.resnet18(pretrained=True).to(self.device) 
        self.layer = self.model._modules.get('avgpool')
        #self.model.eval()
        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        print("device is", self.device)
                                        

    def fit(self, *data):
        pass

    '''

    def transform(self, data):
        to_tensor = transforms.ToTensor()
        #print("enter")
        X_nps = np.array([d['feature'] for d in data.examples])
        #X_nps = np.array([to_tensor(np.repeat(np.pad(d['feature'], ((0,0), (98,98), (98,98))), 3, axis=0)) for d in data.examples])
        n = 20
        #embeddings = torch.zeros(n, 512, 1, 1)
        X_emb = []
       
        #print(type(X_nps))
        #print("shape", X_nps.shape)
        
        for i in range(0, len(X_nps), n):
            #print((i, i+n))
            embeddings = torch.zeros(n, 512, 1, 1)
            tmp = np.pad(X_nps[i:i+n,:,:,:], ((0,0),(0,0), (98,98), (98,98)))
            #print("tmp shape before repeat", tmp.shape)
            tmp = np.repeat(tmp, 3, axis=1)
            #print("tmp shape", tmp.shape)
            X_tensors = self.normalize(torch.from_numpy(tmp).float())
            X_tensors = X_tensors.to(self.device)
            #print(X_tensors.size())
            def copy_data(m, i, o):
                embeddings.copy_(o.data)
            
            hk = self.layer.register_forward_hook(copy_data)
            with torch.no_grad():
                _ = self.model(X_tensors)
            #print(len(embeddings))
            #print(embeddings.size())
            partial = torch.reshape(embeddings, (n,-1))
            #print(partial.size())
            X_emb.extend(partial)
            #quit()
            hk.remove()
        #print(len(X_emb))
        return self._repack_data(data, X_emb)
    '''

    def transform(self, data):
        X_nps = np.array([d['feature'] for d in data.examples])
        n = 20
        X_emb = []
       
        copy_data = self.CopyData()
        for i in range(0, len(X_nps), n):
            #print((i, i+n))
            tmp = np.pad(X_nps[i:i+n,:,:,:], ((0,0),(0,0), (98,98), (98,98)))
            #print("tmp shape before repeat", tmp.shape)
            tmp = np.repeat(tmp, 3, axis=1)
            #print("tmp shape", tmp.shape)
            X_tensors = self.normalize(torch.from_numpy(tmp).float())
            X_tensors = X_tensors.to(self.device)
            
            hk = self.layer.register_forward_hook(copy_data)
            with torch.no_grad():
                _ = self.model(X_tensors)
            hk.remove()
            #print("copy data is", len(copy_data.outputs))
        X_emb = copy_data.outputs
        #print(len(X_emb))
        return self._repack_data(data, X_emb)

    def fit_transform(self):
        pass

    class CopyData:
        def __init__(self):
            self.outputs = []
        
        def __call__(self, module, module_in, module_out):
            self.outputs.extend(module_out)
        
        def clear(self):
            self.outputs = []
    
if __name__ == '__main__':
    ''' Example usage... 
    '''
    from wrench.dataset import load_dataset

    dataset_home = '../../datasets'
    data = 'MNIST'
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, 
        extract_feature=True,
        dataset_type='NumericDataset')

    embedder = PretrainedModelEmbedding()

    # Fit the union of all unlabeled examples
    embedder.fit(train_data, valid_data, test_data)

    train_data = embedder.transform(train_data)
    # print("size of train data embeddings", getsizeof(train_data))
    #quit()
    valid_data = embedder.transform(valid_data)
    print("size of valid data embeddings", getsizeof(valid_data))
    #quit()
    test_data = embedder.transform(test_data)
    print("size of test data embeddings", getsizeof(test_data))

    print(len(test_data.examples[4]['feature']))