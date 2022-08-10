import numpy as np
import torch
from .base_embedding import BaseEmbedding
from tqdm import tqdm
import clip
from PIL import Image
import numpy as np
from .zeroshot_labels import classes_
from .clip_datasets import NLP_DATASETS, IMAGE_DATASETS

class OpenAICLIPEmbedding(BaseEmbedding):
    def __init__(self, dataset, prompt=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load('RN50', self.device)
        self.model.eval()
        self.dataset = dataset
        self.prompt = prompt
        self.context_length = 77 #default clip https://github.com/openai/CLIP

    def extract_label_text_features(self, label_text):
        zeroshot_weights = []
        for label_t in label_text:
            texts = clip.tokenize(label_t).to(self.device)
            class_embeddings = self.model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.device)
        return zeroshot_weights

    def extract_joint_image_text_features(self, X, label_text):
        image_features_all = []
        print("Extracting image features...")
        for image_id in tqdm(range(X.shape[0])):
            image = X[image_id,:,:,:].detach().cpu().numpy()
            image = np.transpose(image, (1,2,0))
            image = Image.fromarray(np.uint8(image*255.))
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_feature = self.model.encode_image(image_input)
            image_feature /= image_feature.norm()
            image_features_all.append(image_feature)
        image_features_all = torch.stack(image_features_all, dim=1).to(self.device)
        image_features_all = image_features_all.squeeze()
        text_features_all = self.extract_label_text_features(label_text)
        logits = (100. * image_features_all @ text_features_all).softmax(dim=-1).detach().cpu()
        return logits

    def fit(self, *data):
        pass
    
    def transform(self, data):
        if self.dataset in NLP_DATASETS:
            self.transform_text(data)
        elif self.dataset in IMAGE_DATASETS:
            self.transform = self.transform_image(data)
    
    def extract_text_feature(self, raw_text):
        text_tokenized = clip.tokenize(raw_text, context_length=self.context_length).to(self.device)
        text_embedding = self.model.encode_text(text_tokenized)
        return text_embedding
    
    def transform_text(self, data):
        X_raw, y = self._unpack_data(data, flatten=False, return_y=True, raw=True)
        text_features_all = []
        for raw_obj in tqdm(X_raw):
            try:
                text_orig = raw_obj['text']
                text = text_orig[:self.context_length] # truncated to context length
                text_embedding = self.extract_text_feature(text)
                text_features_all.append(text_embedding)
            except:
                print(f"EXCEPTION ON TEXT: {text_orig} truncating to 1/10 context length and retrying..")
                text = text[:len(text)//10]
                text_embedding = self.extract_text_feature(text)
                text_features_all.append(text_embedding)
                
        text_features_all = torch.squeeze(torch.stack(text_features_all, dim=1)).to(self.device)
        return self._repack_data(data, text_features_all)
    
    def transform_image(self, data):
        X_np = self._unpack_data(data, flatten=False, return_y=False)
        y = classes_[self.dataset]
        if self.prompt: #promps are assumed to be before the label. e,g., "This is an image of the digit {label}"
            label_text = [f"{self.prompt} {y_}" for y_ in y]
        else:
            label_text = [f"{y_}" for y_ in y]
        if X_np.shape[1] == 1:  # Repeat since MNIST is greyscale
            X_np = X_np.repeat(3, axis=1)
        elif X_np.shape[1] > 3:  # Probably need to permute
            X_np = np.transpose(X_np, (0, 3, 1, 2))
        with torch.no_grad():
            X = torch.from_numpy(X_np)
            X = X.type(torch.FloatTensor)
            X.cuda()
            X_feats = self.extract_joint_image_text_features(X, label_text)
        return self._repack_data(data, X_feats)

    def fit_transform(self, data, ngpus=1, max_epochs=5, hidden_size=128):
        self.fit(data)
        return self.transform(data)
