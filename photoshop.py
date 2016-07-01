import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import tarfile
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from PIL import Image
im=Image.open("./123.png").convert("L")
width,height = im.size
data_image = im.getdata()
d = np.matrix(data_image,dtype = float)
d = np.reshape(data_image,(height,width))/255.0
#matrix converted into image
matrix_shape = d.shape
matrix_row = matrix_shape[0]
matrix_column = matrix_shape[1]
row_res=[]
for i in range(matrix_row):
	i=i+1
	if i==1:
		res = np.sum(d[0:i])
	else:
		res = np.sum(d[(i-1):i])
	row_res.append(res)
#get the image boundary of row
#save the image boundary data of row in One-dimensional array 
row_boundary=[]
#max_member = max(map(lambda x:(row_res.count(x),x),row_res))
for i in range(len(row_res)):
	if i==0:
		a = row_res[i]
	if i!=0 and a != row_res[i]:
		if (len(row_boundary)+1)%2 == 1:
			row_boundary.append(i-1)
			a=row_res[i]
		if (len(row_boundary)+1)%2 ==0 and row_res[0] == row_res[i]:
			row_boundary.append(i+1)
			a=row_res[i]
#partition the matrix by the image boundary of row in order to get each row of image
process_image_number = len(row_boundary)/2
column_res = [[] for t in  range(process_image_number)]
for j in range(len(row_boundary)):
	if j%2 == 0:
		list_to_array = d[row_boundary[j]:row_boundary[(j+1)],:]
		for i in range(list_to_array.shape[1]):
			i=i+1
			if i==1:
				res = np.sum(list_to_array[0:,0:i])
			else:
				res = np.sum(list_to_array[0:,(i-1):i])
			column_res[(j/2)].append(res)
#get the image boundary of column from each row of image
column_boundary = [[] for t in range(len(column_res))]
for j in range(len(column_res)):
	max_member = max(map(lambda x:(column_res[j].count(x),x),column_res[j]))
	for i in range(len(column_res[j])):
		if i==0:
			a = column_res[j][i]
		if i!=0 and a != column_res[j][i]:
			if (len(column_boundary[j])+1)%2 == 1:
				column_boundary[j].append(i-1)
				a=column_res[j][i]
			if (len(column_boundary[j])+1)%2 ==0 and column_res[j][0] == column_res[j][i]:
				column_boundary[j].append(i+1)
				a=column_res[j][i]
#partition the matrix by the char boundary in order to get each image
image_size = 28
image_numbers = 0
for j in range(process_image_number):
	number = len(column_boundary[j])/2
	image_numbers = image_numbers + number

dataset = np.ndarray(shape=(image_numbers, image_size, image_size),dtype=np.float32)
num_image = 0
print row_boundary
print column_boundary
for j in range(len(row_boundary)):
	if j%2 == 0:
		list_to_array = d[row_boundary[j]:row_boundary[(j+1)],:]
		j=j/2
		for i in range(len(column_boundary[j])):
			if i%2 == 0:
				index1 = column_boundary[j][i]		
				index2 = column_boundary[j][(i+1)]
				one_image = list_to_array[0:,index1:index2]
				image_shape = one_image.shape
				if image_shape != (image_size,image_size):
					if image_shape[0] > image_shape[1]:
						res = image_shape[0] - image_shape[1]
						matrix_merger = np.zeros((image_shape[0],res)) + list_to_array[0:1,0:1]
	  					one_image = np.column_stack((one_image,matrix_merger))
						newimage = Image.fromarray(one_image)
						newimage = newimage.resize((image_size,image_size))
						data = newimage.getdata()
						data = np.matrix(data,dtype = float)
						data = np.reshape(data,(image_size,image_size))
						dataset[num_image,:,:] = data
						num_image = num_image+1
					elif image_shape[0] < image_shape[1]:
						res = image_shape[1] - image_shape[0]
						matrix_merger = np.zeros((res,image_shape[1])) + list_to_array[0:1,0:1]
						one_image = np.row_stack((one_image,matrix_merger))
						newimage = Image.fromarray(one_image)
						newimage = newimage.resize((image_size,image_size))
						data = newimage.getdata()
						data = np.matrix(data,dtype = float)
						data = np.reshape(data,(image_size,image_size))
						dataset[num_image,:,:] = data
						num_image = num_image + 1
					else:
						newimage = Image.fromarray(one_image)
						newimage = newimage.resize((image_size,image_size))
						data = newimage.getdata()
						data = np.matrix(data,dtype = float)
						data = np.reshape(data,(image_size,image_size))
						dataset[num_image,:,:] = data
						num_image = num_image + 1
				else:
					dataset[num_image,:,:] = one_image
					num_image = num_image + 1
print num_image

