# import the necessary packages
from skimage import data, io, segmentation, color
from skimage.future import graph
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.segmentation import watershed
from skimage.util import img_as_float
from random import random
from time import time
from functools import partial
from multiprocessing import Pool, Array
from copy import copy
import matplotlib.pyplot as plt
import argparse
import SimpleITK as sitk
import numpy as np
#import pycuda.autoinit
#import pycuda.driver as pcd
#from pycuda.compiler import SourceModule
from . import MITools as mt
from . import CTViewer as cv
try:
	from tqdm import tqdm # long waits are not fun
except:
	print('tqdm not installed')
	tqdm = lambda x : x

NODULE_THRESHOLD = -600

# load the image and convert it to a floating point data type
def normalization(x):
	x = np.array(x, dtype=float)
	Min = np.min(x)
	Max = np.max(x)
	x = (x - Min) / (Max - Min)
	return x

def weight_mean_color(graph, src, dst, n):
	diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
	diff = np.linalg.norm(diff)
	return {'weight': diff}

def merge_mean_color(graph, src, dst):
	graph.node[dst]['total color'] += graph.node[src]['total color']
	graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
	graph.node[dst]['mean color'] = (graph.node[dst]['total color'] / graph.node[dst]['pixel count'])

def cluster_center_filter(volume, coords):
	#fields = np.array([[0,0,0],[0,0,1],[0,0,-1],[0,1,0],[0,1,1],[0,1,-1],[0,-1,0],[0,-1,1],[0,-1,-1],
	#		   [1,0,0],[1,0,1],[1,0,-1],[1,1,0],[1,1,1],[1,1,-1],[1,-1,0],[1,-1,1],[1,-1,-1],
	#		   [-1,0,0],[-1,0,1],[-1,0,-1],[-1,1,0],[-1,1,1],[-1,1,-1],[-1,-1,0],[-1,-1,1],[-1,-1,-1]])
	fields = np.array([[0,0,0]])
	cind = 0
	while cind<len(coords):
		coord = coords[cind]
		for f in range(len(fields)):
			field = coord + fields[f]
			if volume[field[0], field[1], field[2]]<-600:
				coords.pop(cind)
				break
		cind += 1
	return coords

def cluster_filter(volume, labels):
	labels_filtered = labels.copy()
	num_labels = labels.max() + 1	#the default range of labels is 0 to max
	for label in range(num_labels):
		print('filter process:%d/%d' %(label, num_labels))
		clvalues = volume[labels==label]
		if clvalues.size>0 and clvalues.max()<NODULE_THRESHOLD:
			labels_filtered[labels==label] = -1
	return labels_filtered

'''
def filter_by_label(label, volume, labels):
	clvalues = volume[labels == label]
	if clvalues.size > 0 and clvalues.max() < NODULE_THRESHOLD:
		labels[labels == label] = -1
		#return True
	#return False
def cluster_filter_fast(volume, labels):
	#labels_filtered = labels.copy()
	num_labels = labels.max() + 1  # the default range of labels is 0 to max
	v = Array('f', volume)
	l = Array('f', labels)
	filter = partial(filter_by_label, volume=v, labels=l)
	#for label in range(num_labels):
	#	filter(label)
	pool = Pool(4)
	pool.map(filter, range(num_labels))
	pool.close()
	pool.join()
	#eliminate_labels = labels_invalid.nonzero()[0]
	#eliminate_coords = np.where(labels in eliminate_labels)
	#labels_filtered[eliminate_coords] = -1
	return l.values
'''

def cluster_merge(volume, labels):
	g = graph.rag_mean_color(volume, labels)
	labels_merged = graph.merge_hierarchical(labels, g, thresh=0.03, rag_copy=False, in_place_merge=True,
					   merge_func=merge_mean_color, weight_func=weight_mean_color)
	return labels_merged
	
def threshold_mask(segimage, lungmask, threshold=NODULE_THRESHOLD):
	shape = segimage.shape
	tissuemask = np.zeros(shape,dtype=int)
	for i in range(shape[0]):
		for j in range(shape[1]):
			for k in range(shape[2]):
				#Do judge
				if(lungmask[i][j][k]==1 and segimage[i][j][k]>threshold):
					tissuemask[i][j][k]=1
					#order z,y,x
	np.where(tissuemask == 1)
	return tissuemask
    
def cluster_centers(cluster_labels, eliminate_upper_size=-1):
	num_labels = cluster_labels.max() + 1	#the default range of labels is 0 to max
	cluster_sizes = np.zeros(num_labels, dtype=int)
	centersz = np.zeros(num_labels, dtype=float)
	centersy = np.zeros(num_labels, dtype=float)
	centersx = np.zeros(num_labels, dtype=float)
	label_coords = np.where(cluster_labels >= 0)
	print("cluster centering:")
	for lci in tqdm(range(len(label_coords[0]))):
		z = label_coords[0][lci]
		y = label_coords[1][lci]
		x = label_coords[2][lci]
		label = cluster_labels[z][y][x]
		if label>=0 and cluster_sizes[label] >= 0:
			cluster_sizes[label] += 1
			centersz[label] += z
			centersy[label] += y
			centersx[label] += x
			if eliminate_upper_size>0 and cluster_sizes[label] > eliminate_upper_size:
				# no longer to caluculate the center of this cluster for its too large
				cluster_sizes[label] = -1

	labels_valid = cluster_sizes > 0
	clcenters = np.transpose(np.int_([centersz[labels_valid]/cluster_sizes[labels_valid]+0.5, centersy[labels_valid]/cluster_sizes[labels_valid]+0.5, centersx[labels_valid]/cluster_sizes[labels_valid]+0.5]))
	cllabels = labels_valid.nonzero()[0]

	return clcenters, cllabels

def cluster_centers_fast(cluster_labels, eliminate_upper_size=-1, medical_filter=False, volume=None):
	clcenters = []
	cllabels = []
	labels = np.unique(cluster_labels)
	if labels[0]<0:
		labels = np.delete(labels, 0)
	#num_labels = len(labels)
	#maxlabel = labels.max()
	print('num labels:%d' %(len(labels)))
	'''
	ccl = partial(cluster_center_labelwise, cluster_labels=cluster_labels, eliminate_upper_size=eliminate_upper_size, medical_filter=medical_filter, volume=volume)
	#pool = Pool(50)
	#centers = pool.map(ccl, range(num_labels))
	centers = []
	for i in range(num_labels):
		centers.append(ccl(i))
	#print(centers)
	
	centers.append(None)
	centersarray = np.array(centers)
	centervalid = centersarray!=None
	clcenters = centersarray[centervalid]
	cllabels = np.where(centervalid)[0]
	'''

	for label in enumerate(tqdm(labels)):
		#print('centering process:%d/%d' %(label, maxlabel))
		label = label[1]
		cluster_label = cluster_labels == label
		coords = cluster_label.nonzero()
		cluster_size = len(coords[0])
		if medical_filter and volume is not None:
			clvalues = volume[cluster_label]
			if medical_filter and clvalues.size > 0 and clvalues.max() < NODULE_THRESHOLD:
				#the cluster is of all empty tissue, unnecessary to calculate its center
				continue
		coords = (cluster_label).nonzero()
		if eliminate_upper_size>=0 and cluster_size>eliminate_upper_size:
			continue
		if cluster_size>0:
			center = np.int_([coords[0].mean()+0.5, coords[1].mean()+0.5, coords[2].mean()+0.5])
			clcenters.append(center)
			cllabels.append(label)
	
	return clcenters, cllabels
	
def cluster_center_labelwise(label, cluster_labels, eliminate_upper_size=-1, medical_filter=False, volume=None):
	#print('label:{}' .format(label))
	#starttime = time()
	cluster_label = cluster_labels == label
	coords = cluster_label.nonzero()
	cluster_size = len(coords[0])
	#print('init:{}' .format(time()-starttime))
	if cluster_size==0:
		return None
	if eliminate_upper_size>=0 and cluster_size>eliminate_upper_size:
		return None
	if medical_filter and volume is not None:
		clvalues = volume[cluster_label]
		if clvalues.size > 0 and clvalues.max() < NODULE_THRESHOLD:
			#the cluster is of all empty tissue, unnecessary to calculate its center
			del clvalues
			return None
	#print('filter:{}' .format(time()-starttime))
	center = [int(coords[0].mean()+0.5), int(coords[1].mean()+0.5), int(coords[2].mean()+0.5)]
	#print('centering:{}' .format(time()-starttime))
	del cluster_labels
	del cluster_label
	del coords
	if volume is not None:
		del volume
	
	return center
def cluster_centers_parallel(cluster_labels, eliminate_upper_size=-1, medical_filter=False, volume=None):
	#clcenters = []
	#cllabels = []
	#num_labels = cluster_labels.max() + 1	#the default range of labels is 0 to max
	labels = np.unique(cluster_labels)
	if labels[0]<0:
		labels = np.delete(labels, 0)
	print('num labels:%d' %(len(labels)))
	ccl = partial(cluster_center_labelwise, cluster_labels=cluster_labels, eliminate_upper_size=eliminate_upper_size, medical_filter=medical_filter, volume=volume)
	pool = Pool(5)
	centers = pool.map(ccl, [l for l in labels])
	#centers = []
	#for l in labels:
	#	centers.append(ccl(l))
	#print(centers)
	
	centers.append(None)
	centersarray = np.array(centers)
	centervalid = centersarray!=None
	clcenters = centersarray[centervalid]
	labelindices = np.where(centervalid)[0]
	cllabels = [labels[li] for li in labelindices]
	'''
	for label in range(num_labels):
		print('centering process:%d/%d' %(label, num_labels))
		if medical_filter and volume is not None:
			clvalues = volume[cluster_labels == label]
			if clvalues.size > 0 and clvalues.max() < NODULE_THRESHOLD:
				#the cluster is of all empty tissue, unnecessary to calculate its center
				continue
		coords = (cluster_labels==label).nonzero()
		if eliminate_upper_size>=0 and len(coords[0])>eliminate_upper_size:
			continue
		if len(coords[0])>0:
			center = np.int_([coords[0].mean()+0.5, coords[1].mean()+0.5, coords[2].mean()+0.5])
			clcenters.append(center)
			cllabels.append(label)
	'''
	
	return clcenters, cllabels

'''
def cluster_centers_gpu(cluster_labels):
	num_labels = cluster_labels.max() + 1
	volume_gpu = pcd.to_device(cluster_labels.astype(np.int32))
	volume_copy_gpu = pcd.to_device(np.zeros(cluster_labels.shape, dtype=np.int32))
	width_gpu = pcd.to_device(np.int_([cluster_labels.shape[2]]))
	height_gpu = pcd.to_device(np.int_([cluster_labels.shape[1]]))
	depth_gpu = pcd.to_device(np.int_([cluster_labels.shape[0]]))
	zsums_gpu = pcd.to_device(np.zeros(num_labels, dtype=np.float32))
	ysums_gpu = pcd.to_device(np.zeros(num_labels, dtype=np.float32))
	xsums_gpu = pcd.to_device(np.zeros(num_labels, dtype=np.float32))
	vnums_gpu = pcd.to_device(np.zeros(num_labels, dtype=np.int32))
	mod = SourceModule("""
		__global__ void cluster_centers(int *volume, int *volume_copy, int *xlength, int *ylength, int *zlength, float *xsums, float *ysums, float *zsums, int *vnums) {
			for (int xcoord = threadIdx.x; xcoord < (*xlength); xcoord += blockDim.x) {
				for (int ycoord = threadIdx.y; ycoord < (*ylength); ycoord += blockDim.y) {
					for (int zcoord = threadIdx.z; zcoord < (*zlength); zcoord += blockDim.z) {
						int idx = xcoord * (*ylength) * (*zlength) + ycoord * (*zlength) + zcoord;
						int label = volume[idx];
						volume_copy[idx] = label;
						vnums[label] += 1;
						xsums[label] += xcoord;
						ysums[label] += ycoord;
						zsums[label] += zcoord;
					}
				}
			}
		}
	""")
	func = mod.get_function("cluster_centers")
	func(volume_gpu, volume_copy_gpu, depth_gpu, height_gpu, width_gpu, zsums_gpu, ysums_gpu, xsums_gpu, vnums_gpu, block=(10,10,10), grid=(1,1))
	zsums = pcd.from_device(zsums_gpu, num_labels, np.float32)
	ysums = pcd.from_device(ysums_gpu, num_labels, np.float32)
	xsums = pcd.from_device(xsums_gpu, num_labels, np.float32)
	vnums = pcd.from_device(vnums_gpu, num_labels, np.int32)
	volume_copy = pcd.from_device(volume_copy_gpu, cluster_labels.shape, np.int32)
	
	valid_labels = vnums > 0
	valid_vnums = vnums[valid_labels]
	cllabels = valid_labels.nonzero()[0]
	clcenters = np.empty((cllabels.size, 3), dtype=float)
	clcenters[:,0] = zsums[valid_labels]
	clcenters[:,1] = ysums[valid_labels]
	clcenters[:,2] = xsums[valid_labels]
	clcenters = (clcenters / valid_vnums.reshape((valid_vnums.size, 1)) + 0.5).astype(int)
	
	return clcenters, cllabels
	
def cluster_centers_gpu(cluster_labels):
	print("cluster centering")
	labels = np.unique(cluster_labels)
	if labels[0]<0:
		labels = np.delete(labels, 0)
	volume_gpu = pcd.to_device(cluster_labels.astype(np.int32))
	width_gpu = pcd.to_device(np.int_([cluster_labels.shape[2]]))
	height_gpu = pcd.to_device(np.int_([cluster_labels.shape[1]]))
	depth_gpu = pcd.to_device(np.int_([cluster_labels.shape[0]]))
	numlabels_gpu = pcd.to_device(np.int_([labels.size]))
	labels_gpu = pcd.to_device(labels.astype(np.int32))
	sums_gpu = pcd.to_device(np.zeros((len(labels), 3), dtype=np.float32))
	vnums_gpu = pcd.to_device(np.zeros(len(labels), dtype=np.int32))
	mod = SourceModule("""
		__global__ void cluster_centers(int *volume, int *width, int *height, int *depth, int *numlabels, int *labels, float *sums, int *vnums) {
			for (int zcoord = 0; zcoord < (*depth); zcoord ++) {
				for (int ycoord = 0; ycoord < (*height); ycoord ++) {
					for (int xcoord = 0; xcoord < (*width); xcoord ++) {
						for(int index = threadIdx.x; index < (*numlabels); index += blockDim.x) {
							int idx = zcoord * (*height) * (*width) + ycoord * (*width) + xcoord;
							if (labels[index] == volume[idx]) {
								vnums[index] ++;
								sums[3*index] += zcoord;
								sums[3*index+1] += ycoord;
								sums[3*index+2] += xcoord;
							}
						}
					}
				}
			}
		}
	""")
	func = mod.get_function("cluster_centers")
	func(volume_gpu, width_gpu, height_gpu, depth_gpu, numlabels_gpu, labels_gpu, sums_gpu, vnums_gpu, block=(128,1,1), grid=(1,1))
	sums = pcd.from_device(sums_gpu, (len(labels), 3), np.float32)
	vnums = pcd.from_device(vnums_gpu, (len(labels), 1), np.int32)
	clcenters = (sums/vnums+0.5).astype(int)
	print("centering done.")
	
	return clcenters, labels
'''

def cluster_size_vision(labels, outname="clsizes.txt"):
	outfile = open(outname, "w")
	num_labels = labels.max() + 1
	lsizes = np.zeros(shape=[num_labels], dtype=int)
	for z in range(labels.shape[0]):
		for y in range(labels.shape[1]):
			for x in range(labels.shape[2]):
				if labels[z][y][x]>=0:
					lsizes[labels[z][y][x]] += 1
	for lsize in lsizes:
		outfile.write("%d " %(lsize))
	outfile.close()
		

def segment_color_vision(labels):
	random_vision = np.zeros(shape=(labels.shape[0], labels.shape[1], labels.shape[2], 3))
	cluster_colors = np.random.rand(labels.max()+1, 3)
	cluster_coords = np.where(labels>=0)
	for z, y, x in np.nditer([cluster_coords[0], cluster_coords[1], cluster_coords[2]]):
		cind = labels[z, y, x]
		if cind>=0:
			random_vision[z, y, x] = cluster_colors[cind]
	return random_vision

def segment_vision(volume, labels):
	if volume.min()<0 or volume.max()>1:
		volume = normalization(volume)
	segvision = np.zeros(shape=(volume.shape[0], volume.shape[1]*2-1, volume.shape[2]*2-1, 3), dtype=np.float64)
	#cv.view_CT(labels)
	for z in range(volume.shape[0]):
		segvision[z] = mark_boundaries(volume[z], labels[z], mode='subpixel')
	return segvision

def seed_coord_cluster(index, clsize):
	numcoords = len(index[0])
	if numcoords == 0:
		return []
	coords = []
	for i in range(len(index[0])):
		coords.append([index[0][i], index[1][i], index[2][i]])

	clnum = 0
	index_cluster = 0 - np.ones(numcoords, dtype=int)
	steps = [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, 1, 1], [0, 1, -1], [0, -1, 0], [0, -1, 1], [0, -1, -1],
		 [1, 0, 0], [1, 0, 1], [1, 0, -1], [1, 1, 0], [1, 1, 1], [1, 1, -1], [1, -1, 0], [1, -1, 1],
		 [1, -1, -1],
		 [-1, 0, 0], [-1, 0, 1], [-1, 0, -1], [-1, 1, 0], [-1, 1, 1], [-1, 1, -1], [-1, -1, 0], [-1, -1, 1],
		 [-1, -1, -1]]
	# steps = [[0,0,1],[0,0,-1],[0,1,0],[0,-1,0],[1,0,0],[-1,0,0]]
	# clustering by seeds
	clusters = []
	for i in range(0, numcoords):
		if index_cluster[i] < 0:
			print("%d" % (i))
			clnum += 1
			cluster_stack = [i]
			index_cluster[i] = clnum - 1
			clusters.append([i])
			size = 1
			while len(cluster_stack) > 0 and size <= clsize:
				pind = cluster_stack.pop(0)
				for step in steps:
					neighbor = [coords[i][0] + step[0], coords[i][1] + step[1],
						    coords[i][2] + step[2]]
					if coords.count(neighbor) > 0:
						nind = coords.index(neighbor)
						if index_cluster[nind] < 0:
							size += 1
							cluster_stack.append(nind)
							index_cluster[nind] = clnum - 1
							clusters[-1].append(nind)

	# calculate the cluster center
	clind = 0
	clend = False
	clcenters = []
	while index_cluster.count(clind) > 0:
		summary = [0.0, 0.0, 0.0]
		size = 0
		for i in range(len(index_cluster)):
			if index_cluster[i] == clind:
				size += 1
				summary = [summary[0] + coords[i][0], summary[1] + coords[i][1],
					   summary[2] + coords[i][2]]
		center = np.array([round(summary[0] / size), round(summary[1] / size), round(summary[2] / size)],
				  dtype=int)
		clcenters.append(center)

	# the coordination order is z, y, x
	return clcenters

def seed_mask_cluster(nodule_matrix, cluster_size=-1):
	clnum = 0
	index_cluster = 0 - np.ones(nodule_matrix.shape, dtype=int)
	steps = [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, 1, 1], [0, 1, -1], [0, -1, 0], [0, -1, 1], [0, -1, -1],
		 [1, 0, 0], [1, 0, 1], [1, 0, -1], [1, 1, 0], [1, 1, 1], [1, 1, -1], [1, -1, 0], [1, -1, 1],
		 [1, -1, -1],
		 [-1, 0, 0], [-1, 0, 1], [-1, 0, -1], [-1, 1, 0], [-1, 1, 1], [-1, 1, -1], [-1, -1, 0], [-1, -1, 1],
		 [-1, -1, -1]]
	# steps = [[0,0,1],[0,0,-1],[0,1,0],[0,-1,0],[1,0,0],[-1,0,0]]
	# clustering by seeds
	#clusters = []
	for z in range(nodule_matrix.shape[0]):
		for y in range(nodule_matrix.shape[1]):
			for x in range(nodule_matrix.shape[2]):
				#print("%d %d %d" %(z, y, x))
				if nodule_matrix[z][y][x] > 0 and index_cluster[z][y][x] < 0:
					clnum += 1
					cluster_stack = [[z, y, x]]
					index_cluster[z][y][x] = clnum - 1
					#clusters.append([[z, y, x]])
					size = 1
					while len(cluster_stack) > 0:
						coord = cluster_stack.pop(0)
						for step in steps:
							neighbor = np.array([coord[0] + step[0], coord[1] + step[1],
									     coord[2] + step[2]], dtype=int)
							if not mt.coord_overflow(neighbor, nodule_matrix.shape) and \
									nodule_matrix[
										neighbor[0], neighbor[1], neighbor[
											2]] > 0 and index_cluster[
								neighbor[0], neighbor[1], neighbor[2]] < 0:
								size += 1
								cluster_stack.append(neighbor)
								index_cluster[neighbor[0], neighbor[1], neighbor[
									2]] = clnum - 1
								#clusters[-1].append(neighbor)
						if cluster_size > 0 and size > cluster_size:
							break

	# the coordination order is z, y, x
	#return clcenters, index_cluster
	return index_cluster

def seed_volume_cluster(volume, segmask, difference_threshold=20, cluster_size=-1, eliminate_lower_size=-1):
	clnum = 0
	index_cluster = 0 - np.ones(volume.shape, dtype=int)
	steps = [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, 1, 1], [0, 1, -1], [0, -1, 0], [0, -1, 1], [0, -1, -1],
		 [1, 0, 0], [1, 0, 1], [1, 0, -1], [1, 1, 0], [1, 1, 1], [1, 1, -1], [1, -1, 0], [1, -1, 1],
		 [1, -1, -1],
		 [-1, 0, 0], [-1, 0, 1], [-1, 0, -1], [-1, 1, 0], [-1, 1, 1], [-1, 1, -1], [-1, -1, 0], [-1, -1, 1],
		 [-1, -1, -1]]
	# steps = [[0,0,1],[0,0,-1],[0,1,0],[0,-1,0],[1,0,0],[-1,0,0]]
	# clustering by seeds
	#clusters = []
	for z in range(volume.shape[0]):
		for y in range(volume.shape[1]):
			for x in range(volume.shape[2]):
				#print("%d %d %d" %(z, y, x))
				if segmask[z][y][x]>0 and volume[z][y][x] > -600 and index_cluster[z][y][x] == -1:
					clnum += 1
					cluster_stack = [[z, y, x]]
					index_cluster[z][y][x] = clnum - 1
					#clusters.append([[z, y, x]])
					size = 1
					while len(cluster_stack) > 0:
						coord = cluster_stack.pop(0)
						for step in steps:
							neighbor = np.array([coord[0] + step[0], coord[1] + step[1],
									     coord[2] + step[2]], dtype=int)
							if not mt.coord_overflow(neighbor,volume.shape) and segmask[neighbor[0],neighbor[1],neighbor[2]]>0 and volume[neighbor[0],neighbor[1],neighbor[2]]>-600 and index_cluster[neighbor[0],neighbor[1],neighbor[2]]==-1 and abs(volume[coord[0],coord[1],coord[2]]-volume[neighbor[0],neighbor[1],neighbor[2]])<=difference_threshold:
								#need to be modified by A Kun
								size += 1
								cluster_stack.append(neighbor)
								index_cluster[neighbor[0], neighbor[1], neighbor[
									2]] = clnum - 1
								#clusters[-1].append(neighbor)
						if cluster_size > 0 and size > cluster_size:
							break
					if eliminate_lower_size>0 and size<=eliminate_lower_size:
						#eliminate the cluster
						#segmask[index_cluster==clnum-1] = 0
						index_cluster[index_cluster==clnum-1] = -2
						clnum -= 1

	# the coordination order is z, y, x
	return index_cluster

def slic_segment(volume, num_segments=500000, compactness=0.001, merge_cluster=False, result_output=False, view_result=False):
	volume_norm = mt.medical_normalization(volume)
	labels = slic(volume_norm, n_segments=num_segments, sigma=1, multichannel=False, compactness=compactness, slic_zero=True,
		      max_iter=15)

	if merge_cluster:
		merged_labels = cluster_merge(volume_norm, labels)
		if result_output:
			mergeresult = segment_vision(volume, merged_labels)
			np.save('detection_vision/slic_result_merge.npy', mergeresult)
			del mergeresult
		del merged_labels
	'''
	g = graph.rag_mean_color(volume, labels)
	labels_merged = graph.merge_hierarchical(labels, g, thresh=0.03, rag_copy=False, in_place_merge=True,
					   merge_func=merge_mean_color, weight_func=weight_mean_color)
	segresult = np.zeros(shape=(volume.shape[0], volume.shape[1], volume.shape[2], 3), dtype=np.float64)
	for z in range(volume.shape[0]):
		segresult[z] = mark_boundaries(volume[z], labels_merged[z], mode='inner')
	'''
	if result_output or view_result:
		segresult = segment_vision(volume, labels)
		if result_output:
			np.save('detection_vision/slic_result.npy', segresult)
		if view_result:
			cv.view_CT(segresult)

	return labels

def seed_segment(volume, lung_mask, cluster_size=-1, view_result=False):
	organ_mask = threshold_mask(volume, lung_mask)
	labels = seed_volume_cluster(organ_mask, cluster_size)
	labels_merged = cluster_merge(volume, labels)
	if view_result:
		segresult = segment_vision(volume, labels_merged)
		cv.view_CT(segresult)

	return labels_merged