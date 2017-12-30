from mxnet import gluon
from mxnet.gluon.model_zoo.custom_layers import HybridConcurrent

# Helpers
def _make_basic_conv(target = None, **kwargs):
    #print(kwargs['prefix'])
    #exit()
    out = gluon.nn.HybridSequential(prefix='')
    out.add(gluon.nn.Conv2D(use_bias=False, **kwargs))
    if target:
        out.add(gluon.nn.BatchNorm(epsilon=0.001, prefix='0_batchnorm'+target))
    else:
        out.add(gluon.nn.BatchNorm(epsilon=0.001))
    out.add(gluon.nn.Activation('relu'))
    return out

def _make_branch(use_pool, *conv_settings):
    out = gluon.nn.HybridSequential(prefix='')
    if use_pool == 'avg':
        out.add(gluon.nn.AvgPool2D(pool_size=3, strides=1, padding=1))
    elif use_pool == 'max':
        out.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))
    setting_names = ['channels', 'kernel_size', 'strides', 'padding']
    for setting in conv_settings:
        kwargs = {}
        for i, value in enumerate(setting):
            if value is not None:
                kwargs[setting_names[i]] = value
        out.add(_make_basic_conv(**kwargs))
    return out

def _make_A(pool_features, prefix):
    out = HybridConcurrent(concat_dim=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None,
                             (64, 1, None, None)))
        out.add(_make_branch(None,
                             (48, 1, None, None),
                             (64, 5, None, 2)))
        out.add(_make_branch(None,
                             (64, 1, None, None),
                             (96, 3, None, 1),
                             (96, 3, None, 1)))
        out.add(_make_branch('avg',
                             (pool_features, 1, None, None)))
    return out

def _make_B(prefix):
    out = HybridConcurrent(concat_dim=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None,
                             (384, 3, 2, None)))
        out.add(_make_branch(None,
                             (64, 1, None, None),
                             (96, 3, None, 1),
                             (96, 3, 2, None)))
        out.add(_make_branch('max'))
    return out

def _make_C(channels_7x7, prefix):
    out = HybridConcurrent(concat_dim=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None,
                             (192, 1, None, None)))
        out.add(_make_branch(None,
                             (channels_7x7, 1, None, None),
                             (channels_7x7, (1, 7), None, (0, 3)),
                             (192, (7, 1), None, (3, 0))))
        out.add(_make_branch(None,
                             (channels_7x7, 1, None, None),
                             (channels_7x7, (7, 1), None, (3, 0)),
                             (channels_7x7, (1, 7), None, (0, 3)),
                             (channels_7x7, (7, 1), None, (3, 0)),
                             (192, (1, 7), None, (0, 3))))
        out.add(_make_branch('avg',
                             (192, 1, None, None)))
    return out

def _make_D(prefix):
    out = HybridConcurrent(concat_dim=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None,
                             (192, 1, None, None),
                             (320, 3, 2, None)))
        out.add(_make_branch(None,
                             (192, 1, None, None),
                             (192, (1, 7), None, (0, 3)),
                             (192, (7, 1), None, (3, 0)),
                             (192, 3, 2, None)))
        out.add(_make_branch('max'))
    return out

def _make_E(prefix):
    out = HybridConcurrent(concat_dim=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None,
                             (320, 1, None, None)))

        branch_3x3 = gluon.nn.HybridSequential(prefix='')
        out.add(branch_3x3)
        branch_3x3.add(_make_branch(None,
                                    (384, 1, None, None)))
        branch_3x3_split = HybridConcurrent(concat_dim=1, prefix='')
        branch_3x3_split.add(_make_branch(None,
                                          (384, (1, 3), None, (0, 1))))
        branch_3x3_split.add(_make_branch(None,
                                          (384, (3, 1), None, (1, 0))))
        branch_3x3.add(branch_3x3_split)

        branch_3x3dbl = gluon.nn.HybridSequential(prefix='')
        out.add(branch_3x3dbl)
        branch_3x3dbl.add(_make_branch(None,
                                       (448, 1, None, None),
                                       (384, 3, None, 1)))
        branch_3x3dbl_split = HybridConcurrent(concat_dim=1, prefix='')
        branch_3x3dbl.add(branch_3x3dbl_split)
        branch_3x3dbl_split.add(_make_branch(None,
                                             (384, (1, 3), None, (0, 1))))
        branch_3x3dbl_split.add(_make_branch(None,
                                             (384, (3, 1), None, (1, 0))))

        out.add(_make_branch('avg',
                             (192, 1, None, None)))
    return out

def make_aux(classes):
    out = gluon.nn.HybridSequential(prefix='')
    out.add(gluon.nn.AvgPool2D(pool_size=5, strides=3))
    out.add(_make_basic_conv(channels=128, kernel_size=1))
    out.add(_make_basic_conv(channels=768, kernel_size=5))
    out.add(gluon.nn.Flatten())
    out.add(gluon.nn.Dense(classes))
    return out

# Net
class Inception3_modify(gluon.HybridBlock):
    r"""Inception v3 model from
    `"Rethinking the Inception Architecture for Computer Vision"
    <http://arxiv.org/abs/1512.00567>`_ paper.

    Parameters
    ----------
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self, classes=120, **kwargs):
        super(Inception3_modify, self).__init__(**kwargs)
        # self.use_aux_logits = use_aux_logits
        with self.name_scope():
            self.features = gluon.nn.HybridSequential(prefix='')
            self.features.add(_make_basic_conv(channels=32, kernel_size=3, strides=2, prefix='0_conv0_', target='0_'))
            self.features.add(_make_basic_conv(channels=32, kernel_size=3, prefix='0_conv1_', target='1_'))
            self.features.add(_make_basic_conv(channels=64, kernel_size=3, padding=1, prefix='0_conv2_', target='2_'))
            self.features.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))
            self.features.add(_make_basic_conv(channels=80, kernel_size=1, prefix='0_conv3_', target='3_'))
            self.features.add(_make_basic_conv(channels=192, kernel_size=3, prefix='0_conv4_', target='4_'))
            self.features.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))
            self.features.add(_make_A(32, '0_A1_'))
            self.features.add(_make_A(64, '0_A2_'))
            self.features.add(_make_A(64, '0_A3_'))
            self.features.add(_make_B('0_B_'))
            self.features.add(_make_C(128, '0_C1_'))
            self.features.add(_make_C(160, '0_C2_'))
            self.features.add(_make_C(160, '0_C3_'))
            self.features.add(_make_C(192, '0_C4_'))

            self.classifier = gluon.nn.HybridSequential(prefix='')
            self.classifier.add(_make_D('1_D_'))
            self.classifier.add(_make_E('1_E1_'))
            self.classifier.add(_make_E('1_E2_'))
            self.classifier.add(gluon.nn.AvgPool2D(pool_size=8))
            self.classifier.add(gluon.nn.Dropout(0.5))
            self.classifier.add(gluon.nn.Dense(classes, prefix='1_dense0_'))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

