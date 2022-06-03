import numpy as np
import torch
from .base_embedding import BaseEmbedding
from tqdm import tqdm
import clip
from PIL import Image
import numpy as np
from .zeroshot_labels import classes_

class OpenAICLIPEmbedding(BaseEmbedding):
    def __init__(self, dataset, prompt=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load('RN50', self.device)
        self.model.eval()
        self.dataset = dataset
        self.prompt = prompt

    def extract_text_features(self, label_text):
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

    def extract_features(self, X, label_text):
        # text_inputs = torch.cat([clip.tokenize(label_t) for label_t in label_text]).to(self.device)
        # similarity_all = []
        # text_features_all = []
        # image_features_all = []
        # for label_t in label_text:
        #     with torch.no_grad():
        #         text_input = clip.tokenize(label_t).to(self.device)
        #         text_features = self.model.encode_text(text_input)
        #     text_features /= text_features.norm(dim=-1, keepdim=True)
        #     text_features_all.append(text_features)
        # text_features_all = torch.cat(text_features_all)
        
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
        text_features_all = self.extract_text_features(label_text)
        # for x_idx in tqdm(range(X.shape[0])):
        #     x_single = X[x_idx, :, :, :]
        #     x_single = np.uint8(np.transpose(x_single, (1,2,0)))*255
        #     image_input = self.preprocess(Image.fromarray(x_single)).unsqueeze(0).to(self.device)
        #     with torch.no_grad():
        #         image_features = self.model.encode_image(image_input)
        #     image_features /= image_features.norm(dim=-1, keepdim=True)
        #     image_features_all.append(image_features)
        # image_features_all = torch.cat(image_features_all)
        print("IMAGE FEATURES SHAPE", image_features_all.shape)
        logits = (100. * image_features_all @ text_features_all).softmax(dim=-1).detach().cpu()
                # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                # similarity_all.append(similarity.detach().cpu())
        # similarity_all = torch.cat(similarity_all)
        print("LOGIT SHAPE", logits.shape)
        return logits

    def fit(self, *data):
        pass

    def transform(self, data, bs=5120):
        X_np = self._unpack_data(data, flatten=False, return_y=False)
        y = classes_[self.dataset]
        
        if self.prompt: #promps are assumed to be before the label. e,g., "This is an image of the digit {label}"
            label_text = [f"{self.prompt} {y_}" for y_ in y]
        else:
            label_text = [f"{y_}" for y_ in y]
        print(f"OPENAI CLIP ZERO SHOT W/ TEXTS {label_text}")
        if X_np.shape[1] == 1:  # Repeat since MNIST is greyscale
            X_np = X_np.repeat(3, axis=1)
        elif X_np.shape[1] > 3:  # Probably need to permute
            X_np = np.transpose(X_np, (0, 3, 1, 2))
        with torch.no_grad():
            X = torch.from_numpy(X_np)
            X = X.type(torch.FloatTensor)
            X.cuda()
            X_feats = self.extract_features(X, label_text)
        return self._repack_data(data, X_feats)

    def fit_transform(self, data, ngpus=1, max_epochs=5, hidden_size=128):
        self.fit(data)
        return self.transform(data)
