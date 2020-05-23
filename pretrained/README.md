# Download a pretrained weights

Because of Faster RCNN architecture require a shared layers in order to share feature maps to both of Region Proposal Network and Fast RCNN Network for speed efficiency
and this repository can perform a Resnet50 Network or VGG16 Network as a shared layer

So, A shared layer require its own pretained weight that you can be found at following links:

- [resnet50_weights_tf_dim_ordering_tf_kernels.h5](https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5)
- [vgg16_weights_tf_dim_ordering_tf_kernels.h5](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5)

or look further at: https://github.com/fchollet/deep-learning-models/releases

All you need todo is to download a pretrained weight and save it into this directory before perform a training process