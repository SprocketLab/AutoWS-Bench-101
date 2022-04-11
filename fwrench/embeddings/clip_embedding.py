import torch
import numpy as np
from fwrench.embeddings.base_embedding import BaseEmbedding
from transformers import CLIPProcessor, CLIPModel

class ClipEmbedding(BaseEmbedding):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def transform(self, data):
        imgs = np.array([d['feature'] for d in data.examples])
        X_np = []
        image_features = []
        for i in range(len(imgs)):
            #X_np[i] = X_np[i].reshape(int(len(X_np[i])**1/2), -1)
            #X_np[i] = np.stack((X_np[i],)*3, axis=1)
            tmp = np.repeat(imgs[i], 3, axis=0)
            tmp = tmp.transpose((1,2,0))
            #print(tmp.shape)
            X_np.append(tmp)
            if (i+1) % 50 == 0:
                print("i+1: ", i+1)
                inputs = self.processor(images=X_np, return_tensors="pt").to(self.device)
                part_features = self.model.get_image_features(**inputs).to(self.device)
                part_features = part_features.tolist()
                image_features.extend(part_features)
                X_np = []
                #print(len(image_features))

        return self._repack_data(data, image_features)

    def fit(self, *data):
        pass

    '''
    Same as transform
    '''
    def fit_transform(self, data):
        imgs = np.array([d['feature'] for d in data.examples])
        X_np = []
        image_features = []
        for i in range(len(imgs)):
            #X_np[i] = X_np[i].reshape(int(len(X_np[i])**1/2), -1)
            #X_np[i] = np.stack((X_np[i],)*3, axis=1)
            tmp = np.repeat(imgs[i], 3, axis=0)
            tmp = tmp.transpose((1,2,0))
            #print(tmp.shape)
            X_np.append(tmp)
            if (i+1) % 50 == 0:
                print("i+1: ", i+1)
                inputs = self.processor(images=X_np, return_tensors="pt").to(self.device)
                part_features = self.model.get_image_features(**inputs).to(self.device)
                part_features = part_features.tolist()
                image_features.extend(part_features)
                X_np = []
                #print(len(image_features))

        return self._repack_data(data, image_features)

