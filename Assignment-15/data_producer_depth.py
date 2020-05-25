import os
import glob
from torch.utils.data import Dataset
from PIL import Image
import PIL

class DataProducer(Dataset):
    def __init__(self, data_folder, transformForeground=None, transformDepth=None):
        self.data_folder = data_folder
        # self.background_images = list((data_folder / 'bg').glob('*.jpg'))
        self.foreground_images = list((data_folder / 'fg').glob('*.jpg'))
        # self.mask_images = list((data_folder / 'mask').glob('*.jpg'))
        self.depth_images = list((data_folder / 'depth').glob('*.jpg'))
        self.transformForeground = transformForeground
        self.transformDepth = transformDepth
    
    def __len__(self):
        return len(self.depth_images)
    
    def __getitem__(self, index):
        depth_image = Image.open(self.depth_images[index]).convert('L')
        depth_image = PIL.ImageOps.invert(depth_image)
        depth_image = depth_image.resize((64,64))
        fg_image = Image.open(self.get_foreground_image(self.depth_images[index]))
        fg_image = fg_image.resize((64,64))
        # bg_image = Image.open(self.get_background_image(self.foreground_images[index]))
        # bg_image = bg_image.resize((64,64))
        # mask_image = Image.open(self.get_mask_image(self.foreground_images[index])).convert('L')
        # mask_image = mask_image.resize((64,64))
        # Do transformation
        
        if self.transformForeground:
            fg_image = self.transformForeground(fg_image)
            # bg_image = self.transform(bg_image)
            # mask_image = self.transform(mask_image)

        if self.transformDepth:
            depth_image = self.transformDepth(depth_image)
        
        return { 'fg': fg_image, 'depth' : depth_image}
    
    def get_foreground_image(self, depth_image_path):
        # print('File Image Path', fg_image_path)
        file_path = str(self.data_folder) + '/depth/depth_'
        # print('File Path', file_path)
        file_name = str(depth_image_path).replace(file_path,'')
        #str(fg_image_path).lstrip(file_path)
        # index = file_name.find('_')
        # print(file_name)
        # file_name = file_name[:index] + '.jpg'
        return self.data_folder / 'fg' / file_name
    
    # def get_mask_image(self, fg_image_path):
    #     # file_path = str(self.data_folder / 'fg')
    #     file_path = str(self.data_folder) + '/fg/'
    #     file_name = 'mask_' + str(fg_image_path).replace(file_path,'')#str(fg_image_path).lstrip(file_path)
    #     return self.data_folder / 'mask' / file_name