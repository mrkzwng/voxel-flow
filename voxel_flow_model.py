"""Implements a voxel flow model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils.loss_utils import l1_loss, l2_loss, vae_loss, l1_regularizer 
from utils.geo_layer_utils import vae_gaussian_layer
from utils.geo_layer_utils import bilinear_interp
from utils.geo_layer_utils import meshgrid

'''TODO
Check regularization terms again;
- Check Charbonnier penalty function
'''


FLAGS = tf.app.flags.FLAGS

class Voxel_flow_model(object):
  def __init__(self, is_train=True):
    self.is_train = is_train

  def inference(self, input_images):
    """Inference on a set of input_images.
    Args:
    """
    return self._build_model(input_images) 

  def loss(self, predictions, flow_motion, flow_mask, targets, lambda_motion, lambda_mask):
    """Compute the necessary loss for training.
    Args:
    Returns:
    """
    # corrected regularized loss
    self.reproduction_loss = l1_loss(predictions, targets) \
                  # + lambda_motion * l1_regularizer(flow_motion) \
                  # + lambda_mask * l1_regularizer(flow_mask)

    return self.reproduction_loss

  def _build_model(self, input_images):
    """Build a VAE model.
    Args:
    """

    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)):
      
      # Define network     
      batch_norm_params = {
        'decay': 0.9997,
        'epsilon': 0.001,
        'is_training': self.is_train,
      }
      with slim.arg_scope([slim.batch_norm], is_training = self.is_train, updates_collections=None):
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
          normalizer_params=batch_norm_params):
          # encoders
          conv_1 = slim.conv2d(input_images, 16, [5, 5], stride=1, scope='conv1')
          pool_1 = slim.max_pool2d(conv_1, [2, 2], scope='pool1')
          conv_2 = slim.conv2d(pool_1, 32, [5, 5], stride=1, scope='conv2')
          pool_2 = slim.max_pool2d(conv_2, [2, 2], scope='pool2')
          conv_3 = slim.conv2d(pool_2, 64, [3, 3], stride=1, scope='conv3')
          pool_3 = slim.max_pool2d(conv_3, [2, 2], scope='pool3')
          bottleneck = slim.conv2d(pool_3, 128, [3, 3], stride=1, scope='bottleneck')
          # decoders
          upsamp_1 = tf.image.resize_bilinear(bottleneck, [64, 64])
          deconv_1 = slim.conv2d(tf.concat([upsamp_1, conv_3], axis=3), 
                        64, [3, 3], stride=1, scope='deconv4')
          upsamp_2 = tf.image.resize_bilinear(deconv_1, [128, 128])
          deconv_2 = slim.conv2d(tf.concat([upsamp_2, conv_2], axis=3), 
                        32, [3, 3], stride=1, scope='deconv5')
          # upsampled to input dimensions
          upsamp_A = tf.image.resize_bilinear(deconv_2, [256, 256])
          deconv_A = slim.conv2d(tf.concat([upsamp_A, conv_1], axis=3), 
                        32, [5, 5], stride=1, scope='deconvA')
          upsamp_B = tf.image.resize_bilinear(deconv_1, [256, 256])
          deconv_B = slim.conv2d(upsamp_B, 32, [5, 5], stride=1, scope='deconvB')
          upsamp_C = tf.image.resize_bilinear(bottleneck, [256, 256])
          deconv_C = slim.conv2d(upsamp_C, 32, [5, 5], stride=1, scope='deconvC')
          # concatenated
          conv_4 = slim.conv2d(tf.concat([deconv_A, deconv_B, deconv_C], axis=3),
                        64, [5, 5], scope='conv_4')

    net = slim.conv2d(conv_4, 3, [5, 5], stride=1, activation_fn=tf.tanh,
    normalizer_fn=None, scope='conv7')
    net_copy = net
    
    flow = net[:, :, :, 0:2]
    mask = tf.expand_dims(net[:, :, :, 2], 3)

    grid_x, grid_y = meshgrid(256, 256)
    grid_x = tf.tile(grid_x, [FLAGS.batch_size, 1, 1])
    grid_y = tf.tile(grid_y, [FLAGS.batch_size, 1, 1])

    flow = 0.5 * flow

    coor_x_1 = grid_x + flow[:, :, :, 0]
    coor_y_1 = grid_y + flow[:, :, :, 1]

    coor_x_2 = grid_x - flow[:, :, :, 0]
    coor_y_2 = grid_y - flow[:, :, :, 1]    
    
    output_1 = bilinear_interp(input_images[:, :, :, 0:3], coor_x_1, coor_y_1, 'interpolate')
    output_2 = bilinear_interp(input_images[:, :, :, 3:6], coor_x_2, coor_y_2, 'interpolate')

    mask = 0.5 * (1.0 + mask)
    mask = tf.tile(mask, [1, 1, 1, 3])
    net = tf.multiply(mask, output_1) + tf.multiply(1.0 - mask, output_2)

    # for the correct loss function
    flow_motion = net_copy[:, :, :, 0:2]
    flow_mask = tf.expand_dims(net[:, :, :, 2], 3)

    return(net, flow_motion, flow_mask)