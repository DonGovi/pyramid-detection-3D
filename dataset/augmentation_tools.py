from skimage.transform import AffineTransform, rotate, PiecewiseAffineTransform, resize, warp
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import os

class AugMethod2D(object):
	def __init__(self, image):
		"""
		:param image: image should be a 2D picture with size(height,width,channels)
		or if image is a gray image,then it can also be size(height,width), the length of which is 2
		"""
		self.image = image
	
	def flip_lr(self):
		return np.fliplr(self.image)
	
	def flip_ud(self):
		return np.flipud(self.image)
	
	def rotate(self, angle):
		image_rotate = rotate(self.image, angle, resize=True,mode="reflect")
		if image_rotate.shape != self.image.shape:
			image_rotate = resize(image_rotate, self.image.shape, mode="reflect")
		return image_rotate
	
	def aff_trans(self, x_cut=None, y_cut=None):
		if x_cut is None and y_cut is None:
			img_height, img_width, *_ = self.image.shape
			x_cut = -0.15 * img_height
			y_cut = -0.15 * img_width
		afftrans_matrix = np.array([
			[1.2, 0.15, x_cut],
			[0.25, 1.2, y_cut],
			[0, 0, 1]])
		image_aff = warp(self.image, AffineTransform(matrix=afftrans_matrix),mode="reflect")
		return image_aff
	
	@staticmethod
	def sin_aff_trans(image, rescale):
		return None
	
	def show_image(self):
		fig, ax = plt.subplots()
		if len(self.image.shape) == 2:
			ax.imshow(self.image, cmap="gray")
			plt.show()
		else:
			ax.imshow(self.image)
			plt.show()

class AugMethod3D(object):
	def __init__(self, image):
		"""
		:param image: image should be a 3D picture with size(depth,height,width,channels)
		or if image is a gray image,then it can also be size(depth,height,width), the length of which is 2
		or the image can also be a flat image s.t. size(k*height,k*width,channels) or (k*height,k*width)
		"""
		self.image = image
	
	def flip(self, trans_position):
		"""
		:param trans_position: can be one of item in ["D","H","W"], choose a trans position and then
		we will do flip in this position
		:return: the flip_lr image in location
		"""
		if trans_position == "D":
			return self.image[::-1, ...]
		if trans_position == "H":
			return self.image[:, ::-1, ...]
		if trans_position == "W":
			return self.image[:, :, ::-1, ...]
	
	def rotate(self, angle):
		"""
		:param angle: do rotation for each slice and restore them into an image
		:return: image after rotation
		"""
		image_rotated = np.empty(self.image.shape)
		for index, image_slice in enumerate(self.image):
			rotated_slice = rotate(image_slice, angle, resize=True,mode="reflect")
			if rotated_slice.shape != image_slice.shape:
				rotated_slice = resize(rotated_slice, image_slice.shape, mode="reflect")
			image_rotated[index, ...] = rotated_slice
		return image_rotated
	def aff_trans(self, x_cut=None, y_cut=None):
		if x_cut is None and y_cut is None:
			img_height, img_width, *_ = self.image.shape
			x_cut = -0.15 * img_height
			y_cut = -0.15 * img_width
		afftrans_matrix = np.array([
			[1.2, 0.15, x_cut],
			[0.25, 1.2, y_cut],
			[0, 0, 1]])
		image_affined = np.empty(self.image.shape)
		for index, image_slice in enumerate(self.image):
			image_affined[index,...] = warp(image_slice, AffineTransform(matrix=afftrans_matrix),mode="reflect")
		return image_affined
	def trans_flat_2_cube(self, h, w):
		"""
		if image is a flat image stated in __init__,we can turn it into a cube s.t.(k,h,w)
		:return: cube with (d,h,w)
		"""
		H, W, *_ = self.image.shape
		assert H % h == 0 and W % w == 0, print("can't split the flat image into slices,H/h or W/w is not integer")
		d = int((H * W) / (h * w))
		num_row, num_col = int(H / h), int(W / w)
		cube = np.empty((d, h, w, *_))
		for row_i in range(num_row):
			for col_j in range(num_col):
				cube[row_i * num_col + col_j, ...] = self.image[row_i * h:(row_i + 1) * h, col_j * w:(col_j + 1) * w, ...]
		self.image = cube
	
	@staticmethod
	def trans_cube_2_flat(image):
		"""
		:param image: a cube with (d,h,w,channels),we need to return a flat image containing them. the size of image
		can be i in row * j in col,i*j=d,and we will return a flat image with i the largest factor of d in range(sqrtd)
		:return:
		"""
		d, h, w, *_ = image.shape
		#print(d,h,w, *_)
		num_row = [i for i in range(1, int(math.sqrt(d)) + 1) if d % i == 0][-1]
		num_col = int(d / num_row)
		flat_image = np.empty((h * num_row, w * num_col, *_))
		for row_i in range(num_row):
			for col_j in range(num_col):
				flat_image[row_i * h:(row_i + 1) * h, col_j * w:(col_j + 1) * w, ...] = image[row_i * num_col + col_j, ...]
		return flat_image


if __name__ == '__main__':
	
	file_path = "E:/sph_samples/"
	sample_path = os.path.join(file_path, "samples_128/samples/")
	picture_path = os.path.join(file_path, "samples_128/pictures/")
	'''
	file_path = "D:/SPH_data/resample/"
	sample_path = os.path.join(file_path, "resample_data/")
	picture_path = os.path.join(file_path, "resample_picture/")
	'''
	file_list = os.listdir(sample_path)
	for file in file_list:
		if os.path.splitext(file)[1] == ".npy":
			sample_img = np.load(os.path.join(sample_path, file))
			#print(np.max(sample_img), np.min(sample_img))
			
			sample_img = (sample_img + 1000.0) / 1400.0
			#print(np.max(sample_img), np.min(sample_img))
			sample_img = np.clip(sample_img, 0, 1)
			sample_img *= 255
			save_name = os.path.splitext(file)[0]
			flat_img = AugMethod3D.trans_cube_2_flat(sample_img)
			print(flat_img.shape)
			print(np.mean(flat_img))
			#img_gray = cv2.cvtColor(flat_img, cv2.COLOR_BGR2GRAY)  
			cv2.imwrite(os.path.join(picture_path, save_name+".png"), flat_img)
			cv2.destroyAllWindows()
			print("Save %s" % save_name+".png")