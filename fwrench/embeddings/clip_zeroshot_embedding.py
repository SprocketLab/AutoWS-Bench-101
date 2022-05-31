import numpy as np
import torch
from .base_embedding import BaseEmbedding
from transformers import CLIPProcessor, CLIPVisionModel
from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel

# label_text = [str(i) for i in range(10)]


class ZeroShotCLIPEmbedding(BaseEmbedding):
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def get_image_as_list(self, dataset):
        images = []
        for imgs in dataset:
            images.append((imgs.detach().cpu().numpy()))
        return images

    def extract_feature_batch(self, x, label_text):
        with torch.no_grad():
            inputs = self.processor(text=label_text, images=x, return_tensors="pt",
            padding=True)
            outputs = self.model(
                # pixel_values=inputs.pixel_values,
                # input_ids=inputs.input_ids,
                # return_dict=True,
                **inputs
            )
            image_features = outputs['vision_model_output']['pooler_output'].norm(dim=-1, keepdim=True)
            # text_features = outputs['text_model_output']['pooler_output'].norm(dim=-1, keepdim=True)
            # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            return outputs['logits_per_image']

    def fit(self, *data):
        pass

    def transform(self, data, bs=1280):
        X_np, y = self._unpack_data(data, flatten=False, return_y=True)
        y = np.unique(y)
        label_text = [f"{y_}" for y_ in y]
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
