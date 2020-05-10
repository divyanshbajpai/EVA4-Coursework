import os
import glob
import argparse
import matplotlib

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images, scale_up, save_images
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/*.jpg', type=str, help='Input filename or folder.')
parser.add_argument('--output', default='output', type=str, help='Output folder.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

# Input images
inputs = load_images( glob.glob(args.input) )
print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

# Compute results
outputs = predict(model, inputs)

print(outputs.shape)

new_images = scale_up(2,outputs)

print('value', new_images[0])

for i in range(outputs.shape[0]):
    file_name = os.path.join(os.path.join(os.getcwd(),'output'), str(i) + '.png')
    save_images(file_name,np.resize(outputs[i],(1,128,128,1)))
    # print(new_images[i].shape)
    # im = Image.fromarray(np.resize(outputs[i],(128,128)))
    # im = im.convert("L")
    # im.save(file_name)
    # im.close()


#matplotlib problem on ubuntu terminal fix
#matplotlib.use('TkAgg')   

# Display results
viz = display_images(outputs.copy(), inputs.copy())
plt.figure(figsize=(10,5))
plt.imshow(viz)
plt.savefig('test.png')
plt.show()
