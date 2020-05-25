import matplotlib.pyplot as plt
import torchvision
import numpy as np
import skimage

def show(output_tensors, original_tensors, epoch=None, fig_size=(10,10), *args, **kwargs):
    allimage=[]
    print('Outshape',output_tensors.shape," orginial Shape ",original_tensors.shape)
    for i in range(20):
        
        img_set = np.vstack([torchvision.utils.make_grid(original_tensors[i]).permute(1, 2, 0),torchvision.utils.make_grid(output_tensors[i]).permute(1, 2, 0)])
        allimage.append(img_set)
    allimage=np.stack(allimage)
    vm=skimage.util.montage(allimage, multichannel=True, fill=(0,0,0),grid_shape=(4,5))
    plt.figure(figsize=(15, 20))
    plt.imshow(vm)
    plt.axis('off')
    plt.grid('off')
#   try:
#     tensors = tensors.detach().cpu()
#   except:
#     pass
#   grid_tensor = torchvision.utils.make_grid(output_tensors, *args, **kwargs)
#   grid_image = grid_tensor.permute(1,2,0)
#   # plt.figure(fig_size=fig_size)
#   # print(tensors.shape)
#   # print(tensors[0].shape)
#   # plt.imshow(torchvision.utils.make_grid(tensors[0]).permute(1,2,0))
#   # plt.show()
#   f, axarr = plt.subplots(2,2)
#   axarr[0,0].imshow(torchvision.utils.make_grid(original_tensors[0]).permute(1, 2, 0))
#   axarr[0,0].set_title('Epoch : {}'.format(epoch))
#   axarr[0,1].imshow(torchvision.utils.make_grid(output_tensors[0]).permute(1, 2, 0))
#   axarr[1,0].imshow(torchvision.utils.make_grid(original_tensors[1]).permute(1, 2, 0))
#   axarr[1,1].imshow(torchvision.utils.make_grid(output_tensors[1]).permute(1, 2, 0))

# import matplotlib.pyplot as plt
# import torchvision
# import numpy as np
# import skimage
# plt.title('Plots')
#f, axarr = plt.subplots(1,2)
# axarr[0].set_title('Epoch : {}'.format(10))
# axarr[0].imshow(torchvision.utils.make_grid(train_set[0]["fg"]).permute(1, 2, 0))
# axarr[1].imshow(torchvision.utils.make_grid(train_set[0]["depth"]).permute(1, 2, 0))
# plt.axis('off')
# plt.savefig('test.jpg')
# plt.show()
