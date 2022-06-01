import numpy as np
import torch
from .base_embedding import BaseEmbedding
from transformers import CLIPProcessor, CLIPVisionModel
from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel

classes_ = {
    "mnist": [f"{i}" for i in range(10)],
    "cifar10": ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    
}

class ZeroShotCLIPEmbedding(BaseEmbedding):
    def __init__(self, dataset, prompt=None):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.dataset = dataset
        self.prompt = prompt

    def get_image_as_list(self, dataset):
        images = []
        for imgs in dataset:
            images.append((imgs.detach().cpu().numpy()))
        return images

    def extract_feature_batch(self, x, label_text):
        with torch.no_grad():
            inputs = self.processor(text=label_text, images=x, return_tensors="pt",
            padding=True)
            outputs = self.model(**inputs)
            return outputs['logits_per_image']

    def fit(self, *data):
        pass

    def transform(self, data, bs=1280):
        X_np = self._unpack_data(data, flatten=False, return_y=False)
        y = classes_[self.dataset]
        
        if self.prompt: #promps are assumed to be before the label. e,g., "This is an image of the digit {label}"
            label_text = [f"{self.prompt} {y_}" for y_ in y]
        else:
            label_text = [f"{y_}" for y_ in y]
        print(f"CLIP ZERO SHOT W/ TEXTS {label_text}")
        if X_np.shape[1] == 1:  # Repeat since MNIST is greyscale
            X_np = X_np.repeat(3, axis=1)
        elif X_np.shape[1] > 3:  # Probably need to permute
            X_np = np.transpose(X_np, (0, 3, 1, 2))
        with torch.no_grad():
            X = torch.from_numpy(X_np)
            X = X.type(torch.FloatTensor)
            X.cuda()
            X_list = self.get_image_as_list(X)
            X_feats = []
            # manual batching
            print(f"CLIP extractor requires batching... bs = {bs}")
            for batch_start in tqdm(range(0, len(X_list), bs)):
                out_ = self.extract_feature_batch(
                    X_list[batch_start : batch_start + bs], label_text
                )
                X_feats.extend(out_)
        X_feats = np.vstack(X_feats)
        return self._repack_data(data, X_feats)

    def fit_transform(self, data, ngpus=1, max_epochs=5, hidden_size=128):
        self.fit(data)
        return self.transform(data)
