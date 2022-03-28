# Almost as-is from this tutorial: https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c
# TODO: use load_dataset in wrench/dataset/__init__.py
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import sys
from os import walk
from os import path
import json

def get_embedding(img_name):
    print(img_name)
    img = Image.open(img_name)
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    embedding = torch.zeros(512)

    def copy_data(m, i, o):
        embedding.copy_(o.data.reshape(o.data.size(1)))

    hook = layer.register_forward_hook(copy_data)
    model(t_img)
    hook.remove()

    return embedding

#print("args: ", str(sys.argv))
dir_path = sys.argv[1]
if not path.exists(dir_path):
    print("The path to the image directory is incorrect.")
    sys.exit(1)
imgs = next(walk(dir_path), (None, None, []))[2]

model = models.resnet18(pretrained=True)
layer = model._modules.get('avgpool')
model.eval()

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

#cnt = 0
output = []
for i in imgs:
    i = str(dir_path) + i
    embedding = get_embedding(i)
    print(embedding)
    print("shape of embedding", str(embedding.size()))
    output.append(embedding.numpy().tolist())
    # cnt += 1
    # if cnt == 5:
    #     break

with open('embeddings.json', 'w') as f:
    json.dump(output, f)

# with open('embeddings.json', 'r') as f:
#     a = json.load(f)

# print(len(a))