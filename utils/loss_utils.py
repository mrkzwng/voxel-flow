"""Implements various tensorflow loss layer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


def l1_charbonnier(values, epsilon):
  """
  implements the generalized Charbonnier distance
  """
  return(tf.sqrt(tf.reduce_sum(tf.square(values)) + epsilon))

def l1_charbonnier_loss(predictions, targets, epsilon):
  """
  implements loss via generalized Charbonnier
  """
  return(l1_charbonnier(predictions - targets, epsilon))

def l1_loss(predictions, targets):
  """Implements tensorflow l1 loss.
  Args:
  Returns:
  """
  total_elements = (tf.shape(targets)[0] * tf.shape(targets)[1] * tf.shape(targets)[2]
      * tf.shape(targets)[3])
  total_elements = tf.to_float(total_elements)

  loss = tf.reduce_sum(tf.abs(predictions- targets))
  loss = tf.div(loss, total_elements)
  
  return loss

def l1_regularizer(flow):
  '''
  implements L1 regularization for flow in R^[n x m]
  '''
  total_dim = tf.cast(tf.reduce_prod(tf.cast(tf.shape(flow), tf.float32)), tf.int32)
  flow = tf.reshape(flow, shape=[total_dim, 1])
  l1_magnitude = tf.reduce_sum(tf.abs(flow))

  return(l1_magnitude)


def l2_loss(predictions, targets):
  """Implements tensorflow l2 loss, normalized by number of elements.
  Args:
  Returns:
  """
  total_elements = (tf.shape(targets)[0] * tf.shape(targets)[1] * tf.shape(targets)[2]
      * tf.shape(targets)[3])
  total_elements = tf.to_float(total_elements)

  loss = tf.reduce_sum(tf.square(predictions-targets))
  loss = tf.div(loss, total_elements)
  return loss

def tv_loss():
  #TODO
  pass
  
def vae_loss(z_mean, z_logvar, prior_weight=1.0):
  """Implements the VAE reguarlization loss.
  """
  total_elements = (tf.shape(z_mean)[0] * tf.shape(z_mean)[1] * tf.shape(z_mean)[2]
      * tf.shape(z_mean)[3])
  total_elements = tf.to_float(total_elements)

  vae_loss = -0.5 * tf.reduce_sum(1.0 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))
  vae_loss = tf.div(vae_loss, total_elements)
  return vae_loss

def bilateral_loss():
  #TODO
  pass
