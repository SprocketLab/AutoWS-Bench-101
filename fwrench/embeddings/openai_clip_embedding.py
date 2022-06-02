import numpy as np
import torch
from .base_embedding import BaseEmbedding
from tqdm import tqdm
import clip
from PIL import Image
import numpy as np

classes_ = {
    "mnist": [f"a photo of the number {i}" for i in range(10)],
    "spherical_mnist": [f"{i}" for i in range(10)],
    # ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
    "cifar10": ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    "fashion_mnist": ["t-shirt or top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]
}

class OpenAICLIPEmbedding(BaseEmbedding):
    def __init__(self, dataset, prompt=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load('ViT-B/32', self.device)
        self.dataset = dataset
        self.prompt = prompt

    def extract_features(self, X, label_text):
        text_inputs = torch.cat([clip.tokenize(label_t) for label_t in label_text]).to(self.device)
        similarity_all = []
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        print("Extracting features...")
        for x_idx in tqdm(range(X.shape[0])):
            x_single = X[x_idx, :, :, :]
            x_single = np.uint8(np.transpose(x_single, (1,2,0)))*255
            image_input = self.preprocess(Image.fromarray(x_single)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                similarity_all.append(similarity.detach().cpu())
        similarity_all = torch.cat(similarity_all)
        return similarity_all

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
