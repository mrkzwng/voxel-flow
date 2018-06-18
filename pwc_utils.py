
import tensorflow as tf
from utils.geo_layer_utils import bilinear_interp
from utils.geo_layer_utils import meshgrid


def trainable_synthesize_frame(self, net, input_images):
    net_copy = net
    
    # flow = net[:, :, :, 0:2]
    # mask = tf.expand_dims(net[:, :, :, 2], 3)
    # trainable mask, i.e. delta t flow
    flow = net
    mask = tf.Variable(name='flow_t',
    				   shape=[net.shape[0], net.shape[1], net.shape[2], 1],
    				   dtype=tf.float32, 
    				   initializer=tf.contrib.layers.xavier_initializer()
    				   )

    grid_x, grid_y = meshgrid(net.shape[1], net.shape[2])
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
    flow_mask = tf.expand_dims(net_copy[:, :, :, 2], 3)

    return(net, flow_motion, flow_mask)


def reproduction_loss(predictions, flow_motion, flow_mask, targets,
					  lambda_motion, lambda_mask, epsilon, regularize=True):

	if regularize == True:
		loss = l1_charbonnier_loss(predictions, targets, epsilon) \
	                  + lambda_motion * l1_charbonnier(flow_motion, epsilon) \
	                  + lambda_mask * l1_charbonnier(flow_mask, epsilon)
	else:
		loss = l1_charbonnier_loss(predictions, targets, epsilon)

	return(loss)