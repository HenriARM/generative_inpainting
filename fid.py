# example of calculating the frechet inception distance in Keras for cifar10
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from cv2 import cv2
import io
import json

from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.datasets import cifar10
from skimage.transform import resize
import numpy as np
 
def serialize_numpy(arr, file_path):
	np.save(file_path, arr)

def derialize_numpy(file_path):
	return np.load(file_path)

def calc_fid_params(z):
	mu, sigma = z.mean(axis=0), cov(z, rowvar=False)
	return mu, sigma 

# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)

def calc_dataset_score(mu1, sigma1, z2):
	# calculate mean and covariance statistics
	mu1, sigma1 = mu1, sigma1
	mu2, sigma2 = z2.mean(axis=0), cov(z2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	return ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
 
# calculate frechet inception distance
def calc_fid_score(z1, z2):
	# calculate mean and covariance statistics
	mu1, sigma1 = z1.mean(axis=0), cov(z1, rowvar=False)
	mu2, sigma2 = z2.mean(axis=0), cov(z2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	return ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
 
# forward image tensor to get inception z vector
def calc_z_vector(x):
	# prepare inception v3 model (original input_shape=(299,299,3)), x is shape of (batch, H, W ,C)
	model = InceptionV3(include_top=False, pooling='avg', input_shape=x.shape[1:])
	x = x.astype('float32')
	# x = scale_images(x, (299, 299))
	# scale pixels between -1 and 1, sample-wise. 
	images1 = preprocess_input(x)
	# calculate activation (z vector)
	return model.predict(x)


def main():
	a = np.array([1.0, 2.0, 3.0, 4,0])
	np.save('test3.npy', a)
	d = np.load('test3.npy')
	print(a.shape)
	print(d.shape)

if __name__ == "__main__":
    main()