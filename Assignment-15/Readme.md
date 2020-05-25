# Assignment -15

'*No one cared who I was, until I put on the mask*'

It all started with putting masks on people, not because of GDPR but for Image Segmentation and Monocular Depth Estimation.

Wisdom of ages proved right again - **It's not the destination but the journey which matters**

Presented with small mountains of problems; below is mentioned how I scaled and slipped through these:

## Task

Input : 

- Overlayed image = rand(FG)*rand(BG)

Output:  

- Depth of Overlayed images
- Mask of foreground image 

## Data

Assignment- 14 brought in the basic starting kit.

- FG images
- BG Images
- Making the masks through GIMP
- Getting Depth images through [ialhashim/DenseDepth](https://github.com/ialhashim/DenseDepth)

## Approach

As simple as it could be.

1. Test the ideas and theories on playground on small dataset
2. When sufficiently sure, run it on the whole dataset



## Environment

Here came in the first obstacle on how to get through this project with burning up the Laptop and how to remain connected to Colab runtime. Believe it or not, Colab has recently been very jumpy.

**"Divide and Conquer"**

1. Playground

   - Made a small version of dataset on Google Drive. (1000 images) 

   - Setup a 'Playground' environment on the Colab with no modulariztion, reason - changing the seperate .py files calls for runtime restart.

   - Playground + Small_Dataset become the testing ground.

2. [Vast.ai](https://vast.ai)
   - Rented GTX 1080/2080 Ti GPUs on ad-hoc basis to run the workload uninterrupted.



## Image Segmentation

- [Custom Model](https://github.com/divyanshbajpai/EVA4-Coursework/blob/master/Assignment-15/newmodel.py)

  ```
  Total params: 114,624
  Trainable params: 114,624
  Non-trainable params: 0
  ----------------------------------------------------------------
  Input size (MB): 0.05
  Forward/backward pass size (MB): 44.03
  Params size (MB): 0.44
  Estimated Total Size (MB): 44.52
  --------------------------------------
  ```

```Snippit from forward function
x1 = self.convblock1_1(x) # input conv
x2 = self.convblock1_2(x1)
x2 = x1+x2
x3 = self.convblock1_3(x2)
x3 = x2+x3
#x4 = self.pool1(x3)
x5 = self.convblock2_1(x3)
x4_1 = self.convblock2_2(x3)
x5 = x4_1+x5
x6 = self.convblock2_3(x5)
x6 = x5+x6
x7 = self.convblock2_4(x6)
```

- Time Taken: 300 sec/epoch

### Loss Function

1. Original Image

   ![image-20200525132212859](https://github.com/divyanshbajpai/EVA4-Coursework/blob/master/Assignment-15/images/image-20200525132212859.png)

2.  BCELogitloss()

   ![image-20200525132246542](https://github.com/divyanshbajpai/EVA4-Coursework/blob/master/Assignment-15/images/image-20200525132246542.png)

3.  L1Loss()

   ![image-20200525132701845](https://github.com/divyanshbajpai/EVA4-Coursework/blob/master/Assignment-15/images/image-20200525132701845.png)

4.  BCELoss()

   ![image-20200525133154545](https://github.com/divyanshbajpai/EVA4-Coursework/blob/master/Assignment-15/images/image-20200525133154545.png)

   

### Strategy

- Loss Function : BCELogitloss() - It showed the most promising results
- LR Strategy: 0.1 for 150 epochs and ReducelronPlateau with factor of 0.1 and patience of 5 for further 150 epochs





## Depth Prediction

- [Model](https://github.com/divyanshbajpai/EVA4-Coursework/blob/master/Assignment-15/DepthModel.py)

  - UNet for Brain Segmentation [Link](https://github.com/mateuszbuda/brain-segmentation-pytorch) 

  ```
  Total params: 7,763,041
  Trainable params: 7,763,041
  Non-trainable params: 0
  ----------------------------------------------------------------
  Input size (MB): 0.05
  Forward/backward pass size (MB): 25.25
  Params size (MB): 29.61
  Estimated Total Size (MB): 54.91
  ```

```Snippit of forward function
enc1 = self.encoder1(x)
enc2 = self.encoder2(self.pool1(enc1))
enc3 = self.encoder3(self.pool2(enc2))
enc4 = self.encoder4(self.pool3(enc3))

bottleneck = self.bottleneck(self.pool4(enc4))

dec4 = self.upconv4(bottleneck)
dec4 = torch.cat((dec4, enc4), dim=1)
dec4 = self.decoder4(dec4)
dec3 = self.upconv3(dec4)
dec3 = torch.cat((dec3, enc3), dim=1)
dec3 = self.decoder3(dec3)
dec2 = self.upconv2(dec3)
dec2 = torch.cat((dec2, enc2), dim=1)
dec2 = self.decoder2(dec2)
dec1 = self.upconv1(dec2)
dec1 = torch.cat((dec1, enc1), dim=1)
dec1 = self.decoder1(dec1)
return torch.sigmoid(self.conv(dec1))
```

### Loss Function Results

1. Original Depth Image

   ![Original](https://github.com/divyanshbajpai/EVA4-Coursework/blob/master/Assignment-15/images/image-20200525015932109.png)

2. L1Loss()

   ![image-20200525022003185](https://github.com/divyanshbajpai/EVA4-Coursework/blob/master/Assignment-15/images/image-20200525022003185.png)

    

3.  SmoothL1Loss()

![image-20200525125946094](https://github.com/divyanshbajpai/EVA4-Coursework/blob/master/Assignment-15/images/image-20200525125946094.png)

4. BCEWithLogitsLoss()

   ![image-20200525130601130](https://github.com/divyanshbajpai/EVA4-Coursework/blob/master/Assignment-15/images/image-20200525130601130.png)

### Strategy

- Loss function : L1Loss() giving most accurate results on the playground
- LR Strategy: LR at 0.1 for first 150 epochs and ReducelronPlateau with factor 0.1 and patience of 5 for next 150 epochs 

### Results



1. Segmentation - even rows shows predicted images [Validate.ipynb](https://github.com/divyanshbajpai/EVA4-Coursework/blob/master/Assignment-15/val.ipynb)

![image-20200525161741721](https://github.com/divyanshbajpai/EVA4-Coursework/blob/master/Assignment-15/images/image-20200525161741721.png)

2. Depth Prediction - even rows show the prediction

   ![Pa](https://github.com/divyanshbajpai/EVA4-Coursework/blob/master/Assignment-15/images/image-20200525162006164.png)





## Transformations

As referred from the paper: [depth-map-prediction-from-a-single-image-using-a-multi-scale-deep-network](https://papers.nips.cc/paper/5539-depth-map-prediction-from-a-single-image-using-a-multi-scale-deep-network.pdf) , I tried to introduce the given transformation in my training set, but the results got worse, initial RCA suggest that my dataset had much of outdoor images rather than Indoor which NYU dataset consists of. In the end went ahead with normalization only.

![image-20200525162517000](https://github.com/divyanshbajpai/EVA4-Coursework/blob/master/Assignment-15/images/image-20200525162517000.png)



## Target Architecture

https://www.researchgate.net/publication/332429592_Depth_Estimation_and_Semantic_Segmentation_from_a_Single_RGB_Image_Using_a_Hybrid_Convolutional_Neural_Network

![image-20200525165739000](https://github.com/divyanshbajpai/EVA4-Coursework/blob/master/Assignment-15/images/image-20200525165739000.png)

This was the reference architecture which I was aiming for. But didn't worked out. Implemented few iterations of architecture but kept failing to get the right backprop technique and combination of different losses. 



## Aims

- Have better data management strategies. With High batch sizes CUDA used to run out of memory.
- To reduce the the depth prediction model size and make it under  1 million parameters.
- With high volume of data, real time logging and leveraging tensor-board and logging features to stay connected with processes.

## Learnings

- Read many Research paper and their implementations on arvix.org and paperswithcode.com - An exercise which I haven't done so far. This gave me a glance what all is possible and how untrodden the DL path is. It excited me and gave me numerous ideas to implement and try out. This inspiration is the biggest learning for me for this assignment.
- Managing the infrastructure from scratch and implementing everything from end to end - dataset generation to reporting
- Breaking out of shell, challenging what I knew and pushing my limits to explore 
- This Assignment was a 'torch' which let me into a cave of opportunities of exciting ideas and research

