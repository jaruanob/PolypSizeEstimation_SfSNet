import os, csv
import pickle
import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.font_manager

from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from skimage.transform import resize
from sklearn.metrics import mean_squared_error

#### CAMERA ###
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np

from camera_models import *

from skimage.measure import regionprops, label
from sklearn.metrics import pairwise_distances
from skimage.morphology import opening
from scipy import ndimage as ndi

# To generate test data
from skimage import data
from skimage.filters import sobel


STREL_4 = np.array([[0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]], dtype=bool)

def get_border_image(region):
    convex_hull_mask = region.convex_image
    eroded_image = ndi.binary_erosion(convex_hull_mask, STREL_4, border_value=0)
    border_image = np.logical_xor(convex_hull_mask, eroded_image)
    return border_image

def get_region_diameters(img):

    assert img.dtype == bool and len(img.shape) == 2

    label_img = label(img, connectivity=img.ndim)

    for region in regionprops(label_img):
        border_image = get_border_image(region)
                
        perimeter_coordinates = np.transpose(np.nonzero(border_image))
        pairwise_distances_matrix = pairwise_distances(perimeter_coordinates)
        i, j = np.unravel_index(np.argmax(pairwise_distances_matrix), pairwise_distances_matrix.shape)
        ptA, ptB = perimeter_coordinates[i], perimeter_coordinates[j]
        region_offset = np.asarray([region.bbox[0], region.bbox[1]])
        ptA += region_offset
        ptB += region_offset
        yield pairwise_distances_matrix[i, j], ptA, ptB, border_image

def sortingFiles(arr, n):
    arr.sort(key=lambda x: (len(x), x))
    return arr
  
dpi_res = 100
matplotlib.rcParams["figure.dpi"] = dpi_res
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['xtick.major.pad']='10'
matplotlib.rcParams['ytick.major.pad']='10'
#matplotlib.rcParams['ztick.major.pad']='8'
fontScale = 25
fig_size=(10,10)

path_db = './SyntheticDatabase_testingset_PolypSize/'

list_files = os.listdir(path_db)
list_sorted_files = sortingFiles(list_files,len(list_files))

gt_size_arr = []
vid_name_arr = []
img_name_arr = []

size_lesser_5 = []
size_5_10 = []
size_10_15 = []
size_larger_15 = []
list_vids = os.listdir(path_db) 
list_vids= sortingFiles(list_vids,len(list_vids)) 

idx_vid = 0
plt.ion()
for name_vid in list_vids:
    print('IDX',idx_vid+1,'--------',name_vid)
    
    path_vid = path_db+name_vid+'/'
    path_vid_img = path_vid+'img/'
    path_vid_depth = path_vid+'z/'
    path_vid_mask = path_vid+'mask/'

    list_files = os.listdir(path_vid_img)
    list_files= sortingFiles(list_files,len(list_files))  

    for name_img in list_files:
        print('-'*10,name_vid,'-'*10)
        print(name_img)
        img = cv2.imread(path_vid_img + name_img)
        
        depth = cv2.imread(path_vid_depth + name_img)[:,:,0]
        polyp_mask = cv2.imread(path_vid_mask + name_img)
        
        dim = (1280,1080)
        
        polyp_mask = cv2.resize(polyp_mask, dim, interpolation = cv2.INTER_AREA)
        
        kernel = np.ones((8,8),np.uint8)
        #
        polyp_mask = cv2.morphologyEx(polyp_mask, cv2.MORPH_CLOSE, kernel)
        
        polyp_mask = cv2.erode(polyp_mask,kernel,iterations = 1)
        polyp_mask_resized = cv2.cvtColor(polyp_mask,cv2.COLOR_BGR2GRAY)
        ret,polyp_mask_thres = cv2.threshold(polyp_mask_resized,127,255,0)
        polyp_mask_thres = opening(polyp_mask_thres, STREL_4)
        polyp_mask_bin = polyp_mask_thres > 127

        
        for distance, ptA, ptB, border_image in get_region_diameters(polyp_mask_bin):
            x1, x2, y1, y2 = ptA[1], ptB[1], ptA[0], ptB[0]
        
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)  
        vect_size = img.shape[0]*img.shape[1]
        
        img_norm = img.astype(np.uint8)/255
        img_flatten = img_norm.flatten().reshape(vect_size,3)
        
        x_retro = []
        y_retro = []
        x_retro=[ptA[0],ptB[0]]
        y_retro=[ptA[1],ptB[1]]


        vect_size = 2 # 2 points of the polyp contour
        cx = np.repeat(img.shape[0]/2,vect_size)
        cy = np.repeat(img.shape[1]/2,vect_size)
    
        focal_length_x = 448.13
        focal_length_x = np.repeat(focal_length_x,vect_size)

        focal_length_y = 378.11
        focal_length_y = np.repeat(focal_length_y,vect_size)
        
        x_over_z = (cx - x_retro) / focal_length_x
        y_over_z = (cy - y_retro) / focal_length_y
        
        x, y = ogrid[0:img.shape[0], 0:img.shape[1]]
        
        #---DEPTH ------------------------------------------------------
        
        depth_norm = (25/256)*resize(depth, (1080,1280),  preserve_range=True,
                                    mode='reflect', anti_aliasing=True )
   
        depth_flatten = np.array([depth_norm[ptA[0],ptA[1]], depth_norm[ptB[0],ptB[1]]])
        
        #---- Retroproyection --------------------------

        z_proj = depth_flatten / np.sqrt(1. + x_over_z**2 + y_over_z**2)
        x_proj = x_over_z * z_proj
        y_proj = y_over_z * z_proj
        
        size = round(np.sqrt((x_proj[0] - x_proj[1])**2 + (y_proj[0] - y_proj[1] )**2 + (z_proj[0] - z_proj[1] )**2)*10,1)
        
        print('POLYP SIZE -  Ground truth size:', round(size,2),' mm ')
        vid_name_arr.append(name_vid)
        img_name_arr.append(name_img)
        gt_size_arr.append(size)
        
        if size <= 5: size_lesser_5.append(size)
        if size > 5 and size <= 10: size_5_10.append(size)
        if size > 10 and size <= 15: size_10_15.append(size)
        if size > 15: size_larger_15.append(size)
        
        if False:
            fig = plt.figure()
            plt.imshow(cv2.resize(img, dim, interpolation = cv2.INTER_AREA))
            x1, x2, y1, y2 = ptA[1], ptB[1], ptA[0], ptB[0]
            plt.plot([x1, x2], [y1, y2], color='k', linestyle='--', linewidth=2)
            idx_mask = cv2.findContours(polyp_mask_thres,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            x_points=[]
            y_points=[]
            for idx_point  in idx_mask[0][0]:
                x_points.append(idx_point[0][0])
                y_points.append(idx_point[0][1]) 

            plt.plot(x_points, y_points, color='k', linestyle='-', linewidth=2)
            plt.annotate('Ground truth: '+str(round(size,2))+' mm', xy =(100,100), xytext =(100,100), fontsize=20, weight='bold')
            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.show(block=False)
            input('     Continuar')
            plt.close('all')
        # continue
        
    idx_vid += 1

print('size_lesser_5:',len(size_lesser_5))
print('size_5_10:',len(size_5_10))
print('size_10_15:',len(size_10_15))
print('size_larger_15:',len(size_larger_15)) 
 
# field names 
fields = ['video', 'image', 'size'] 
   
# data rows of csv file 
rows = []
for idx in range(0,len(gt_size_arr)):
    rows.append([vid_name_arr[idx],img_name_arr[idx],gt_size_arr[idx]])

np.savetxt("ground_truth.csv",
		rows,
		delimiter =",",
		fmt ='% s')