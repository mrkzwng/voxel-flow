"""Train a voxel flow model on ucf101 dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dataset
from utils.prefetch_queue_shuffle import PrefetchQueue
import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.data as dat
from datetime import datetime
import random
from random import shuffle
from voxel_flow_model import Voxel_flow_model
from utils.image_utils import imwrite
from functools import partial
import pdb

# directories
train_image_dir = '../results/train/'
test_image_dir = '../results/test/'
checkpoint = './voxel_flow_checkpoints/iter_20'

# hack due to version differences
tf.data = dat

# Define necessary FLAGS
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', checkpoint,
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_image_dir', train_image_dir,
			   """Directory where to output images.""")
tf.app.flags.DEFINE_string('test_image_dir', test_image_dir,
			   """Directory where to output images.""")
tf.app.flags.DEFINE_string('subset', 'train',
                           """Either 'train' or 'validation'.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', checkpoint,
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")
tf.app.flags.DEFINE_integer('max_steps', 10000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 8, 'The number of samples in each batch.')
tf.app.flags.DEFINE_float('initial_learning_rate', 0.00000001,
                          """Initial learning rate.""")
# added regularization parameters
tf.app.flags.DEFINE_float('lambda_motion', 0.01, 
            """Regularization term for motion components""")
tf.app.flags.DEFINE_float('lambda_mask', 0.005, 
            """Regularization term for time component""")
tf.app.flags.DEFINE_float('epsilon', 0.001,                                         
            """Charbonnier distance parameter""")


def _read_image(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_image(image_string, channels=3)
  image_decoded.set_shape([256, 256, 3])
  return tf.cast(image_decoded, dtype=tf.float32) / 127.5 - 1.0

def train(dataset_frame1, dataset_frame2, dataset_frame3):
  """Trains a model."""
  with tf.Graph().as_default():
    # Create input.
    data_list_frame1 = dataset_frame1.read_data_list_file()
    dataset_frame1 = tf.data.Dataset.from_tensor_slices(tf.constant(data_list_frame1))
    dataset_frame1 = dataset_frame1.repeat().shuffle(buffer_size=1000000, seed=1).map(_read_image)
    dataset_frame1 = dataset_frame1.prefetch(100)

    data_list_frame2 = dataset_frame2.read_data_list_file()
    dataset_frame2 = tf.data.Dataset.from_tensor_slices(tf.constant(data_list_frame2))
    dataset_frame2 = dataset_frame2.repeat().shuffle(buffer_size=1000000, seed=1).map(_read_image)
    dataset_frame2 = dataset_frame2.prefetch(100)

    data_list_frame3 = dataset_frame3.read_data_list_file()
    dataset_frame3 = tf.data.Dataset.from_tensor_slices(tf.constant(data_list_frame3))
    dataset_frame3 = dataset_frame3.repeat().shuffle(buffer_size=1000000, seed=1).map(_read_image)
    dataset_frame3 = dataset_frame3.prefetch(100)

    batch_frame1 = dataset_frame1.batch(FLAGS.batch_size).make_initializable_iterator()
    batch_frame2 = dataset_frame2.batch(FLAGS.batch_size).make_initializable_iterator()
    batch_frame3 = dataset_frame3.batch(FLAGS.batch_size).make_initializable_iterator()
    
    # Create input and target placeholder.
    input_placeholder = tf.concat([batch_frame1.get_next(), batch_frame3.get_next()], 3)
    target_placeholder = batch_frame2.get_next()

    # input_resized = tf.image.resize_area(input_placeholder, [128, 128])
    # target_resized = tf.image.resize_area(target_placeholder,[128, 128])

    # Prepare model.
    model = Voxel_flow_model(is_train=True)
    prediction, flow_motion, flow_mask, prediction128, prediction64 = model.inference(input_placeholder)
    target_64 = tf.image.resize_bilinear(target_placeholder, [64, 64])
    target_128 = tf.image.resize_bilinear(target_placeholder, [128, 128])
    # reproduction_loss, prior_loss = model.loss(prediction, target_placeholder)
    reproduction_loss = model.loss(prediction, flow_motion, 
                              flow_mask, target_placeholder,
                              FLAGS.lambda_motion, FLAGS.lambda_mask, 
                              FLAGS.epsilon)
    mod128_loss = model.coarse_loss(prediction128, target_128, FLAGS.epsilon) 
    mod64_loss = model.coarse_loss(prediction64, target_64, FLAGS.epsilon)
    coarse_loss = mod128_loss + mod64_loss
    # total_loss = reproduction_loss + prior_loss
    total_loss = reproduction_loss + coarse_loss
    
    # Perform learning rate scheduling.
    learning_rate = FLAGS.initial_learning_rate

    # Create an optimizer that performs gradient descent.
    opt = tf.train.AdamOptimizer(learning_rate)
    grads = opt.compute_gradients(total_loss)
    update_op = opt.apply_gradients(grads)

    # Create summaries
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    summaries.append(tf.summary.scalar('total_loss', total_loss))
    summaries.append(tf.summary.scalar('reproduction_loss', reproduction_loss))
    summaries.append(tf.summary.scalar('downsample_128_loss', mod128_loss))
    summaries.append(tf.summary.scalar('downsample_64_loss', mod64_loss))
    # summaries.append(tf.summary.scalar('prior_loss', prior_loss))
    summaries.append(tf.summary.image('Input Image (before)', input_placeholder[:, :, :, 0:3], 3))
    summaries.append(tf.summary.image('Input Image (after)', input_placeholder[:, :, :, 3:6], 3))
    summaries.append(tf.summary.image('Output Image', prediction, 3))
    summaries.append(tf.summary.image('Target Image', target_placeholder, 3))
    # summaries.append(tf.summary.image('Flow', flow, 3))

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge_all()

    # Restore checkpoint from file.
    if FLAGS.pretrained_model_checkpoint_path \
        and tf.train.get_checkpoint_state(FLAGS.pretrained_model_checkpoint_path):
      sess = tf.Session()
      assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
      ckpt = tf.train.get_checkpoint_state(
               FLAGS.pretrained_model_checkpoint_path)
      restorer = tf.train.Saver()
      restorer.restore(sess, ckpt.model_checkpoint_path)
      print('%s: Pre-trained model restored from %s' %
        (datetime.now(), ckpt.model_checkpoint_path))
      sess.run([batch_frame1.initializer, batch_frame2.initializer, batch_frame3.initializer])
    else:
      # Build an initialization operation to run below.
      print('No existing checkpoints.')
      init = tf.global_variables_initializer()
      sess = tf.Session()
      sess.run([init, batch_frame1.initializer, batch_frame2.initializer, batch_frame3.initializer])

    # Summary Writter
    summary_writer = tf.summary.FileWriter(
      FLAGS.train_dir,
      graph=sess.graph)

    data_size = len(data_list_frame1)
    epoch_num = int(data_size / FLAGS.batch_size)

    for step in range(0, FLAGS.max_steps):
      batch_idx = step % epoch_num
      
      # Run single step update.
      _, loss_value = sess.run([update_op, total_loss])
      
      if batch_idx == 0:
        print('Epoch Number: %d' % int(step / epoch_num))
      
      if step % 10 == 0:
        print("Loss at step %d: %f" % (step, loss_value))

      if step % 10 == 0:
        # Output Summary 
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      if step % 50 in {0, 1, 2}:
        # Run a batch of images 
        prediction_np, target_np = sess.run([prediction, target_placeholder])
        for i in range(0,prediction_np.shape[0]):
          file_name = FLAGS.train_image_dir+'_step'+str(step)+'_out.png'
          file_name_label = FLAGS.train_image_dir+'_step'+str(step)+'_gt.png'
          imwrite(file_name, prediction_np[i,:,:,:])
          imwrite(file_name_label, target_np[i,:,:,:])

      # Save checkpoint 
      if step % 50 == 0 or (step +1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

def validate(dataset_frame1, dataset_frame2, dataset_frame3):
  """Performs validation on model.
  Args:  
  """
  pass

def test(dataset_frame1, dataset_frame2, dataset_frame3):
  """Perform test on a trained model."""
  with tf.Graph().as_default():
		# Create input and target placeholder.
    input_placeholder = tf.placeholder(tf.float32, shape=(None, 256, 256, 6))
    target_placeholder = tf.placeholder(tf.float32, shape=(None, 256, 256, 3))
    
    # input_resized = tf.image.resize_area(input_placeholder, [128, 128])
    # target_resized = tf.image.resize_area(target_placeholder,[128, 128])

    # Prepare model.
    model = Voxel_flow_model(is_train=False)
    prediction, flow_motion, flow_mask = model.inference(input_placeholder)
    # reproduction_loss, prior_loss = model.loss(prediction, target_placeholder)
    reproduction_loss = model.loss(prediction, flow_motion,
            flow_mask, target_placeholder, 
            FLAGS.lambda_motion, FLAGS.lambda_mask,
            FLAGS.epsilon)
    # total_loss = reproduction_loss + prior_loss
    total_loss = reproduction_loss

    # Create a saver and load.,
    saver = tf.train.Saver(tf.all_variables())
    sess = tf.Session()

    # Restore checkpoint from file.
    if FLAGS.pretrained_model_checkpoint_path \
            and tf.train.get_checkpoint_state(FLAGS.pretrained_model_checkpoint_path):
      assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
      ckpt = tf.train.get_checkpoint_state(
               FLAGS.pretrained_model_checkpoint_path)
      restorer = tf.train.Saver()
      restorer.restore(sess, ckpt.model_checkpoint_path)
      print('%s: Pre-trained model restored from %s' %
        (datetime.now(), ckpt.model_checkpoint_path))
    else:
      print('No existing model checkpoint in '+checkpoint)
      os.exit(0)
    
    # Process on test dataset.
    data_list_frame1 = dataset_frame1.read_data_list_file()
    data_size = len(data_list_frame1)
    epoch_num = int(data_size / FLAGS.batch_size)

    data_list_frame2 = dataset_frame2.read_data_list_file()

    data_list_frame3 = dataset_frame3.read_data_list_file()

    i = 0 
    PSNR = 0


    """ TODO
    Replace below dataset_frame*.process_func lines with lines 65-84 in 'train'
    """


    for id_img in range(0, data_size):  
      # Load single data.
      line_image_frame1 = dataset_frame1.process_func(data_list_frame1[id_img])
      line_image_frame2 = dataset_frame2.process_func(data_list_frame2[id_img])
      line_image_frame3 = dataset_frame3.process_func(data_list_frame3[id_img])
      
      batch_data_frame1 = [dataset_frame1.process_func(ll) for ll in data_list_frame1[0:63]]
      batch_data_frame2 = [dataset_frame2.process_func(ll) for ll in data_list_frame2[0:63]]
      batch_data_frame3 = [dataset_frame3.process_func(ll) for ll in data_list_frame3[0:63]]
      
      batch_data_frame1.append(line_image_frame1)
      batch_data_frame2.append(line_image_frame2)
      batch_data_frame3.append(line_image_frame3)
      
      batch_data_frame1 = np.array(batch_data_frame1)
      batch_data_frame2 = np.array(batch_data_frame2)
      batch_data_frame3 = np.array(batch_data_frame3)
      
      feed_dict = {input_placeholder: np.concatenate((batch_data_frame1, batch_data_frame3), 3),
                    target_placeholder: batch_data_frame2}
      # Run single step update.
      prediction_np, target_np, loss_value = sess.run([prediction,
                                                      target_placeholder,
                                                      total_loss],
                                                      feed_dict = feed_dict)
      print("Loss for image %d: %f" % (i,loss_value))
      file_name = FLAGS.test_image_dir+str(i)+'_out.png'
      file_name_label = FLAGS.test_image_dir+str(i)+'_gt.png'
      imwrite(file_name, prediction_np[-1,:,:,:])
      imwrite(file_name_label, target_np[-1,:,:,:])
      i += 1
      PSNR += 10*np.log10(255.0*255.0/np.sum(np.square(prediction_np-target_np)))
    print("Overall PSNR: %f db" % (PSNR/len(data_list)))
      
if __name__ == '__main__':
  
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"

  if FLAGS.subset == 'train':
    
    data_list_path_frame1 = "data_list/frame1.txt"
    data_list_path_frame2 = "data_list/frame2.txt"
    data_list_path_frame3 = "data_list/frame3.txt"
    
    ucf101_dataset_frame1 = dataset.Dataset(data_list_path_frame1) 
    ucf101_dataset_frame2 = dataset.Dataset(data_list_path_frame2) 
    ucf101_dataset_frame3 = dataset.Dataset(data_list_path_frame3) 
    
    train(ucf101_dataset_frame1, ucf101_dataset_frame2, ucf101_dataset_frame3)
  
  elif FLAGS.subset == 'test':
    
    data_list_path_frame1 = "data_list/frame1.txt"
    data_list_path_frame2 = "data_list/frame2.txt"
    data_list_path_frame3 = "data_list/frame3.txt"
    
    ucf101_dataset_frame1 = dataset.Dataset(data_list_path_frame1) 
    ucf101_dataset_frame2 = dataset.Dataset(data_list_path_frame2) 
    ucf101_dataset_frame3 = dataset.Dataset(data_list_path_frame3) 
    
    test(ucf101_dataset_frame1, ucf101_dataset_frame2, ucf101_dataset_frame3)
