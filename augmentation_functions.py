import scipy.ndimage as ndi
import numpy as np
import tensorflow as tf
import cv2


# def random_shear(x, intensity,fill_mode='mirror', cval=0., seed=1):
	
# 	np.random.seed(seed)
# 	shear = np.pi / 180 * np.random.uniform(-intensity, intensity)
# 	shear_matrix = np.array([[1, -np.sin(shear), 0],
#                              [0, np.cos(shear), 0],
#                              [0, 0, 1]])

# 	h, w = x.shape[0], x.shape[1]
# 	transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
# 	x = apply_transform(x, transform_matrix, fill_mode, cval)
# 	return x

# def random_rotation(x, intensity,fill_mode='mirror', cval=0., seed=1):

# 	np.random.seed(seed)
# 	theta = np.pi / 180 * np.random.uniform(-intensity, intensity)
# 	rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
# 	                            [np.sin(theta), np.cos(theta), 0],
# 	                            [0, 0, 1]])

# 	h, w = x.shape[0], x.shape[1]
# 	transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
# 	x = apply_transform(x, transform_matrix, fill_mode, cval)
# 	return x


# def random_zoom(x, intensity, fill_mode='nearest', cval=0., seed=1):
    
#     np.random.seed(seed)
#     zoom_range = (1-intensity, 1.0)
#     if zoom_range[0] == 1 and zoom_range[1] == 1:
#         zx, zy = 1, 1
#     else:
#         zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
#     zoom_matrix = np.array([[zx, 0, 0],
#                             [0, zy, 0],
#                             [0, 0, 1]])

#     h, w = x.shape[0], x.shape[1]
#     transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
#     x = apply_transform(x, transform_matrix,  fill_mode, cval)
#     return x

# intensity and seed bogus values
def flip_axis(x, intensity, seed, axis=1):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def random_brightness(x, intensity,seed=1):

	np.random.seed(seed)
    # Convert 2 HSV colorspace from BGR colorspace
	hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    # Generate new random brightness
	rand = np.random.uniform(1-intensity, 1+intensity)
	hsv[:, :, 2] = np.clip(rand*hsv[:, :, 2], a_min=0, a_max=255)
    # Convert back to BGR colorspace
	new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	return new_img
	#return tf.image.random_brightness(x, intensity,seed=seed)

def random_saturation(x, intensity,seed=1):

	np.random.seed(seed)
    # Convert 2 HSV colorspace from BGR colorspace
	hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    # Generate new random brightness
	rand = np.random.uniform(1-intensity, 1+intensity)
	hsv[:, :, 1] = np.clip(rand*hsv[:, :, 1], a_min=0, a_max=255)
    # Convert back to BGR colorspace
	new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	return new_img

def random_hue(x, intensity,seed=1):

	np.random.seed(seed)
    # Convert 2 HSV colorspace from BGR colorspace
	hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    # Generate new random brightness
	rand = np.random.uniform(1-intensity, 1+intensity)
	hsv[:, :, 0] = np.clip(rand*hsv[:, :, 0], a_min=0, a_max=255)
    # Convert back to BGR colorspace
	new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	return new_img

''' Utilities for applying affine transformations (zoom, shear, rotate etc...) '''

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, 2, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, 2+1)
    return x