import os

from scipy.spatial.distance import euclidean

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image

# create the model and find the layer we want
# to derive feature vectors for
model = models.inception_v3(pretrained=True)
model = model.cuda()
# tf slim's inception implementation has a
# named prelogits layer. torchvision's does
# the pooling across the mixed_7c layers as
# an anonymous op. so if we want the output
# from that prelogits layer we need to get 
# the mixed_7c output and then do the 
# pooling ourselves.
layer = model._modules.get('Mixed_7c')
# this is where we'll copy the prelogits
prelogits = torch.zeros(2048)
# add a hook that copies the result of the 
def copy_prelogits(m, i, o):
    # pool the result of the mixed_7c
    # inception layers
    v = F.avg_pool2d(o.data, kernel_size=8)
    v = v.squeeze()
    prelogits.copy_(v.data)

# install the copy data hook
hook = layer.register_forward_hook(copy_prelogits)
# we're not training
model.eval()

# standard transformations for inception
scaler = transforms.Scale((299, 299))
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
to_tensor = transforms.ToTensor()
composed = transforms.Compose([
    scaler,
    to_tensor,
    normalize,
])

# given an image, return the "prelogits"
def extract_prelogits_fv(image_path):
    img = Image.open(image_path)
    img = composed(img).unsqueeze(0)
    img = img.cuda()
    # run inference
    model(img)
    # copy the prelogits out of
    # our placeholder
    return prelogits.clone().numpy()

def main():
    mine = "mine.jpg"
    my_vector = extract_prelogits_fv(mine)

    vectors = {}
    img_dir = "/data-ssd/alex/taxon_photos/61908"
    for file in sorted(os.listdir(img_dir)):
        if "jpg" not in file:
            continue
        img_path = os.path.join(img_dir, file)
        vectors[file] = extract_prelogits_fv(img_path)

    for file in vectors.keys():
        #print(vectors[file])
        distance = euclidean(my_vector, vectors[file])
        print("{} distance to mine.jpg: {}".format(file, distance))

if __name__ == "__main__":
    main()
