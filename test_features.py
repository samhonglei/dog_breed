import mxnet as mx
import numpy as np
import Inception3_modify
from mxnet import gluon, image, nd
from mxnet.gluon.model_zoo import vision as models
import utils
import h5py

'''
h5f = h5py.File('features/train_resnet152_v1.h5', 'r')  # train_resnet152_v1  train_inceptionv31
h5f2 = h5py.File('features/train_inceptionv31.h5', 'r')  # train_resnet152_v1  train_inceptionv31
features1 = h5f['features']
features2 = h5f2['features']

features = np.concatenate([features1[0:10], features2[0:10]], axis=-1)
print(features1[0:10].shape)
exit()
'''
preprocessing = [
    image.ForceResizeAug((224,224)),
    image.ColorNormalizeAug(mean=nd.array([0.485, 0.456, 0.406]), std=nd.array([0.229, 0.224, 0.225]))
]

def transform(data, label):
    data = data.astype('float32') / 255
    for pre in preprocessing:
        data = pre(data)
    data = nd.transpose(data, (2,0,1))
    return data, nd.array([label]).asscalar().astype('float32')

#ctx = [mx.gpu(i) for i in range(1)]
ctx = mx.gpu()
data_dir = './data'
batch_size = 16
epochs = 2

###################################### true train data
#train_imgs = gluon.data.vision.ImageFolderDataset(data_dir+'/train_valid_test/Images', transform=transform)#Images
test_imgs = gluon.data.vision.ImageFolderDataset(data_dir+'/train_valid_test/test', transform=transform)

#train_data = gluon.data.DataLoader(train_imgs, batch_size)    # _dataset  _batch_sampler
test_data = gluon.data.DataLoader(test_imgs, batch_size)

#for batch in (train_data):
    #print(batch)
    #exit()


def save_h5_append(file_name, key, data, batch, l_shape=''):
    h5f = h5py.File(file_name, 'a')
    dataset = h5f[key]
    if l_shape:
        dataset.resize((dataset.shape[0] + batch), axis = 0)
    else:
        dataset.resize([dataset.shape[0]+batch, ])
    dataset[-batch:] = data
    h5f.close()

def save_h5_first(file_name, key, data, l_shape=''):
    h5f = h5py.File(file_name, 'w')
    if l_shape:
        dataset = h5f.create_dataset(key, data=data, maxshape=(None, l_shape))
    else:
        dataset = h5f.create_dataset(key, data=data, maxshape=(None, ))
    h5f.close()

####################   fine-tune gluon

''''''
# resnet152  100352
# inceptionV3  110592
# 获取inceptionV3特征
pretrained_net = models.get_model('inceptionv3', pretrained=True, ctx=ctx)

i = 0
for li_data, label in test_data:
    feature = pretrained_net.features(li_data.as_in_context(ctx))   #<NDArray 16x768x12x12 @gpu(0)>
    feature = gluon.nn.Flatten()(feature)   #<NDArray 16x110592 @gpu(0)>
    if(i==0):
        save_h5_first('test_inceptionv3.h5', 'features', feature.asnumpy(), feature.shape[1])
        #save_h5_first('labels1.h5', 'labels', label.asnumpy())
        i+=1
    else:
        save_h5_append('test_inceptionv3.h5', 'features', feature.asnumpy(), feature.shape[0], feature.shape[1])
        #save_h5_append('labels1.h5', 'labels', label.asnumpy(), label.shape[0])

