from albumentations import *
from albumentations.pytorch import ToTensor
import numpy as np

class TransformationsCust():
  def __init__(self):
    self.test_transform = Compose([OneOf([
		CLAHE(4),
		GridDistortion(p=1),
		OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)],p=1),
		Rotate(5),
		RandomContrast(p=1),
		Flip(p=0.5),

#      Normalize(
#        mean=[0.485,0.456,0.406],
#        std=[0.229,0.224,0.225],
#      ),
      ToTensor()
    ])

  def __call__(self, img):
    img = np.array(img)
    img = self.test_transform(image = img)['image']
    return img