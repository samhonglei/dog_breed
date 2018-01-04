import mxnet as mx
import numpy as np
from mxnet import gluon, image, nd
from mxnet.gluon import nn
from mxnet import autograd
import utils
import h5py
import random
import os

''''''
all_data_count = 20580
range_data = list(range(0,20580))
random.shuffle(range_data)
#print(range_data[0])
#exit()


def accuracy(output, labels):
    return nd.mean(nd.argmax(output, axis=1) == labels).asscalar()

def evaluate(net, data_iter):
    loss, acc, n = 0., 0., 0.
    steps = len(data_iter)
    for data, label in data_iter:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)
        output = net(data)
        acc += accuracy(output, label)
        loss += nd.mean(softmax_cross_entropy(output, label)).asscalar()
    return loss/steps, acc/steps

ctx = mx.gpu()

net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dropout(0.5))
    net.add(nn.Dense(120))

net.load_params('ttt.params', ctx=ctx)
#net.initialize(ctx=ctx)
#net.initialize(init = init.Xavier(),ctx=ctx)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 1e-4, 'wd': 1e-5})


################################### predict ###################################
h5f5 = h5py.File('features/test_resnet152_v1.h5', 'r')  # train_resnet152_v1  train_inceptionv31
h5f6 = h5py.File('features/test_inceptionv3.h5', 'r')  # train_resnet152_v1  train_inceptionv31
features3 = h5f5['features']
features4 = h5f6['features']

train_imgs = gluon.data.vision.ImageFolderDataset( './data/train_valid_test/Images')
ids = sorted(os.listdir('./data/train_valid_test/test/unknown'))
#print(ids)
#exit()
test_count = 10357
outputs = []
for i in range(test_count):
    features = np.concatenate([features3[i:i+1], features4[i:i+1]], axis=-1)
    predict = net(nd.array(features).as_in_context(ctx))
    output = nd.softmax(predict)
    #print(output)
    #exit()
    outputs.extend(output.asnumpy())

with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_imgs.synsets) + '\n')
    for i, output in zip(ids, outputs):
        f.write(i.split('.')[0] + ',' + ','.join([str(num) for num in output]) + '\n')

'''
#################################### train ####################################
#zong shuju  20580
h5f = h5py.File('features/train_resnet152_v1.h5', 'r')  # train_resnet152_v1  train_inceptionv31
h5f2 = h5py.File('features/train_inceptionv31.h5', 'r')  # train_resnet152_v1  train_inceptionv31
h5f3 = h5py.File('features/labels1.h5', 'r') 

features1 = h5f['features']
features2 = h5f2['features']
labels = h5f3['labels']

#features = np.concatenate([features1[0:10], features2[0:10]], axis=-1)
#print(features1[1])
#exit()

#train 
epochs = 3
batch_size = 32
train_split = 18522 #18522

val_features = []
val_labels = []
#range_data
for i in range(train_split, all_data_count):
    range1 = range_data[i]
    val_features.append(np.concatenate([features1[range1], features2[range1]], axis=-1))
    val_labels.append(labels[range1])

dataset_val = gluon.data.ArrayDataset(nd.array(val_features), nd.array(val_labels))
data_iter_val = gluon.data.DataLoader(dataset_val, batch_size)

for epoch in range(epochs):
    train_loss = 0.
    train_acc = 0.
    train_loss_batch = 0.
    train_acc_batch = 0.
    
    max_batch = (int(train_split/batch_size))
    for i in range(max_batch+1):
        
        start_poch = batch_size*i
        if(i-1 == max_batch):
            end_poch = train_split
        else:
            end_poch = batch_size*(i+1)

        train_features = []
        train_labels = []
        range1 = range_data[start_poch:end_poch]
        for ran in range1:
            train_features.append(np.concatenate([features1[ran], features2[ran]], axis=-1))
            train_labels.append(labels[ran])
        
        data, label = nd.array(train_features).as_in_context(ctx), nd.array(train_labels).as_in_context(ctx)
        
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        
        loss.backward()
        trainer.step(batch_size)
        
        train_loss_batch = nd.mean(loss).asscalar()
        train_loss += train_loss_batch
        train_acc_batch = accuracy(output, label)
        train_acc += train_acc_batch
        if not i % 20:
            print("batch %d. loss: %.4f, acc: %.4f%% " % ( 
                i+1,  train_loss_batch, train_acc_batch))
    
    val_loss, val_acc = evaluate(net, data_iter_val)
    print("Epoch %d. loss: %.4f, acc: %.2f%%, val_loss %.4f, val_acc %.2f%%" % (
        epoch+1,  train_loss/max_batch, train_acc/max_batch*100, val_loss, val_acc*100))

net.save_params('ttt.params')
'''