import numpy as np
import torch
import torchvision.models as models


def predict(img: np.ndarray):
    vgg16 = models.vgg16(pretrained=True)
    tensor = torch.from_numpy(img)
    with torch.no_grad():
        prediction = np.argmax(vgg16(tensor))
    return prediction

