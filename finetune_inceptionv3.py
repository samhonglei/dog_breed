import mxnet as mx
import numpy as np
import os
import Inception3_modify
from mxnet import gluon, image, nd, ndarray, init
from mxnet.gluon.model_zoo import vision as models
import sys
sys.path.append('..')
import utils


train_augs = [
    image.HorizontalFlipAug(.5),
    image.RandomCropAug((299,299))
]

test_augs = [
    image.CenterCropAug((299,299))
]

def transform(data, label, augs):
    data = data.astype('float32')
    for aug in augs:
        data = aug(data)
    data = nd.transpose(data, (2,0,1))
    return data, nd.array([label]).asscalar().astype('float32')

ctx = mx.gpu()
data_dir = './data'
batch_size = 16
epochs = 1

###################################### true train data
train_imgs = gluon.data.vision.ImageFolderDataset(
    data_dir+'/train_valid_test/Images',
    transform=lambda X, y: transform(X, y, train_augs))
test_imgs = gluon.data.vision.ImageFolderDataset(
    data_dir+'/train_valid_test/test',
    transform=lambda X, y: transform(X, y, test_augs))

train_data = gluon.data.DataLoader(train_imgs, batch_size, shuffle=True)    # _dataset  _batch_sampler
test_data = gluon.data.DataLoader(test_imgs, batch_size, shuffle=False, last_batch='keep')
#print(vars(train_data._batch_sampler._sampler))
#exit()
#for i, batch in enumerate(train_data):
    #print(batch)
    #exit()
'''

###################################### test_image  最小图片
''''''
min_imgs = gluon.data.vision.ImageFolderDataset(
    data_dir+'/test_image',
    transform=lambda X, y: transform(X, y, train_augs))
train_data = gluon.data.DataLoader(min_imgs, batch_size, shuffle=True)
test_data = train_data

#for X, y in train_data:
    #print(y)
    #X = X.transpose((0,2,3,1)).clip(0,255)/255
    #utils.show_images(X, 2, 2)
    #break
#exit()
#####################################
'''

def train(net, ctx, train_data, test_data, batch_size=64, epochs=1, learning_rate=0.001, wd=0.001):

    # 确保net的初始化在ctx上
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    # 训练
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': wd})
    utils.train(train_data, test_data, net, loss, trainer, ctx, epochs, 10)


####################  init fine-tune
'''
# 判断load params 和 model params一致
load_name = []
for k, v in ndarray.load('inception_v3-0000.params').items():
    key, name = k.split(':')
    load_name.append(name)

cllects = []
for xxx in net.collect_params():
    cllects.append(xxx)

new_collect = []
for collect in cllects:
    #for load in load_name:
    if collect not in load_name:
        new_collect.append(collect)
new_load = []
for load in load_name:
    #for load in load_name:
    if load not in cllects:
        new_load.append(load)

print(new_collect)
print(new_load)
exit()
'''

net = Inception3_modify.Inception3_modify(prefix='inception3')
net.load_params('inception_v31-0000.params', ctx=ctx)
'''
###################################### train ######################################
train(net, ctx, train_data, test_data, batch_size, epochs)
# export
net.export('inception_v31')
'''

'''
###################################### predict ######################################

ids = sorted(os.listdir(os.path.join(data_dir, './train_valid_test/test/unknown')))
#show predict
i = 0 
for data, label in test_data:
    out = net(data.as_in_context(ctx))
    out = nd.SoftmaxActivation(out)
    
    for o in out:
        idx = int(nd.argmax(o, axis=0).asnumpy()[0])
        prob = o[idx]
        name = train_imgs.synsets[idx]
        print(ids[i], prob.asnumpy()[0], idx, name)
        i+=1
exit()

#output csv
outputs = []
for data, label in test_data:
    output = nd.softmax(net(data.as_in_context(ctx)))
    outputs.extend(output.asnumpy())


with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_imgs.synsets) + '\n')
    for i, output in zip(ids, outputs):
        f.write(i.split('.')[0] + ',' + ','.join([str(num) for num in output]) + '\n')
'''
