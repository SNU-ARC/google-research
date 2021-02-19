import numpy as np
import h5py
import os
import requests
import tempfile
import time
import scann

file = "/arc-share/MICRO21_ANNA/GLOVE/glove-100-angular.hdf5"
glove = h5py.File(file, "r")
dataset = glove['train'][0:20, 0:10]
query = glove['test'][0][0:10]
normalized_dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]
vector1 = normalized_dataset[1]
vector2 = normalized_dataset[2]
vector3 = normalized_dataset[3]

print()
c_center = np.array([-0.183827, -0.323822, 0.277935, 0.0409205, 0.105393, -0.207621, 0.298516, -0.249864, -0.0484287, 0.0809632])
p_center = np.array([[0.0840382, 0.157279, 0.0701021, -0.290224, -0.143916, 0.334854, 0.146594, 0.026015, -0.474966, 0.189176, 0.52137, 0.265606, 0.137189, 0.597726, -0.0981454, -0.346124, 0.22284, 0.60531, -0.20342, -0.243161, -0.282513, -0.441167, 0.316489, -0.116093, -0.238653, 0.0770447, 0.633345, 0.187394, -0.152113, 0.107652, 0.226219, 0.0964043], [-0.451814, -0.137407, -0.0786877, -0.186563, 0.308225, -0.153954, -0.346859, -0.333277, -0.356898, 0.191392, 0.359886, -0.0088962, -0.145414, -0.539705, 0.266988, -0.43568, 0.423468, 0.164997, 0.332346, 0.392374, -0.185068, 0.074668, 0.192725, -0.128965, 0.221539, -0.140864, 0.212879, -0.20385, -0.180989, -0.171598, -0.00150796, 0.143009], [-0.00906864, -0.121169, -0.0249646, 0.259174, 0.429316, -0.0278029, -0.516261, -0.411274, 0.263203, 0.0920212, -0.201551, -0.489531, -0.813693, 0.109382, -0.24418, -0.0889431, -0.212042, -0.671123, -0.0794098, -0.339475, -0.0539508, -0.83017, -0.490947, -0.417641, 0.23878, -0.157792, -0.589993, 0.0563982, 0.0459195, -0.335728, 0.0742848, 0.136484], [-0.0308016, -0.00192243, 0.36297, 0.218896, -0.163968, -0.142104, -0.271104, -0.142242, 0.00701205, 0.349118, -0.345115, -0.226869, 0.211338, -0.19785, -0.12661, 0.607698, -0.601709, -0.135374, 0.1587, -0.412084, 0.395208, 0.107854, -0.109417, -0.413388, -0.0839807, 0.0749112, -0.254131, 0.0685184, 0.0743262, 0.0845912, -0.0217078, -0.242464], [0.098146, -0.115262, 0.18219, -0.507419, -0.167054, -0.103445, 0.404568, -0.740367, -5.09119e-05, 0.499086, 0.255477, 0.343917, -0.412329, -0.231154, -0.320831, 0.524337, -0.191884, 0.208766, -0.0684105, -0.315559, 0.429944, 0.246817, -0.0600826, -0.492955, -0.169238, -0.661937, 0.336995, -0.274907, -0.198146, 0.0681149, -0.0380372, -0.442551]])

vector1 = vector1 - c_center
vector2 = vector2 - c_center
vector3 = vector3 - c_center

real_dist = np.array([])
for i in range(dataset.shape[0]):
	real_dist = np.append(real_dist, sum(query*normalized_dataset[i]))
print("real dist")
print(real_dist)

#print(vector1[0], vector1[1])
#print(vector2[0], vector2[1])

#vector1 = vector1 - c_center
#vector2 = vector2 - c_center

index1 = np.array([])
index2 = np.array([])
index3 = np.array([])
for i in  range(p_center.shape[0]):
	arr1 = np.array([])
	arr2 = np.array([])
	arr3 = np.array([])
	for j in range(p_center.shape[1] // 2):
		arr1 = np.append(arr1, sum(vector1[2*i:2*i+2]*p_center[i][2*j:2*j+2]))
		arr2 = np.append(arr2, sum(vector2[2*i:2*i+2]*p_center[i][2*j:2*j+2]))
		arr3 = np.append(arr3, sum(vector3[2*i:2*i+2]*p_center[i][2*j:2*j+2]))
	index1 = np.append(index1, np.argmin(arr1))
	index2 = np.append(index2, np.argmin(arr2))
	index3 = np.append(index3, np.argmin(arr3))
print("index1")
print(index1)
print("index2")
print(index2)
print("index3")
print(index3)

#index = np.array([0, 0, 0, 0, 0])
#dist = 100
#for i1 in range(16):
#	for i2 in range(16):
#		for i3 in range(16):
#			for i4 in range(16):
#				for i5 in range(16):
#					sumq = sum(query[0:2]*p_center[0][2*i1:2*i1+2]) + sum(query[2:4]*p_center[1][2*i2:2*i2+2]) + sum(query[4:6]*p_center[2][2*i3:2*i3+2]) + sum(query[6:8]*p_center[3][2*i4:2*i4+2]) + sum(query[8:10]*p_center[4][2*i5:2*i5+2])
#					if(abs(sumq + 0.631748) < dist):
#						dist = abs(sumq + 0.631748)
#						index = np.array([i1, i2, i3, i4, i5])
#print("index")
#print(index)

#sumq = sum(query[0:2]*p_center[0][2*index[0]:2*index[0]+2]) + sum(query[2:4]*p_center[1][2*index[1]:2*index[1]+2]) + sum(query[4:6]*p_center[2][2*index[2]:2*index[2]+2]) + sum(query[6:8]*p_center[3][2*index[3]:2*index[3]+2]) + sum(query[8:10]*p_center[4][2*index[4]:2*index[4]+2])
#print(sumq)

searcher_path = "/arc-share/MICRO21_ANNA/scann_searcher/glove/Split_1/glove_searcher"

#searcher = scann.scann_ops_pybind.builder(normalized_dataset, 3, "dot_product").tree(
#    num_leaves=4, num_leaves_to_search=2, training_sample_size=20).score_ah(
#    2, anisotropic_quantization_threshold=0.2).reorder(10).build()

#os.makedirs(searcher_path, exist_ok=True)
#searcher.serialize(searcher_path)

searcher = scann.scann_ops_pybind.load_searcher(searcher_path)

neighbors, distances = searcher.search(query)
print("neighbors")
print(neighbors)
print("distances")
print(distances)
