import skimage.io
import skimage.feature
import skimage.morphology
import skimage.filters
from skimage.color import label2rgb
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import cv2
import imagecodecs
import numpy as np
from skimage.morphology import binary_closing, disk
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk


from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)



filepath='F:/EVOS/F.d8.500.0005.tif'
#filepath='F:/EVOS/F.d8.0000.0016.tif'
#filepath='F:/EVOS/F.d8.800.0008.tif'



image = cv2.imread(filepath) 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize=(8, 8)) 
claheNorm = clahe.apply(gray) 
cv2.imwrite(filepath+'claheNorm.tif', claheNorm) 




fig=plt.figure(figsize=(50, 20))

img = skimage.io.imread(filepath+'claheNorm.tif')
img = img[:,:]




thresholds = filters.threshold_multiotsu(img, classes=5)
regions = np.digitize(img, bins=thresholds)

cells = img > thresholds[0]
deadcells = img > thresholds[2]


selem =  morphology.disk(7)
res = morphology.white_tophat(deadcells, selem)


highconfluency=deadcells ^ res
lowconfluency=deadcells


cells=highconfluency




distance = ndi.distance_transform_edt(cells)

local_maxi = feature.peak_local_max(distance, indices=False,
                                    min_distance=1)

markers = measure.label(local_maxi)

segmented_cells = segmentation.watershed(-distance, markers, mask=cells)








underlay = cv2.imread(filepath) 

# remove artifacts connected to image border
cleared = clear_border(segmented_cells)

# label image regions
label_image = label(cleared)


image_label_overlay = label2rgb(label_image, image=underlay, bg_label=0,alpha=0.4)




fig, ax = plt.subplots(2,2, figsize=(20, 16))
ax[0,0].imshow(image, cmap='gray')
ax[0,0].set_title('Basic EVOS')
ax[0,0].axis('off')
#ax[1].imshow(color.label2rgb(segmented_cells, bg_label=0))
ax[0,1].imshow(cells)
ax[0,1].set_title('Dead Cells')
ax[0,1].axis('off')
ax[1,0].imshow(color.label2rgb(segmented_cells, bg_label=0))
ax[1,0].set_title('Watershed Segmentation (Euclidian Distance)')
ax[1,0].axis('off')
ax[1,1].imshow(image_label_overlay)
ax[1,1].set_title('Overlay')
ax[1,1].axis('off')
plt.show()


fig.savefig('C:/Users/Administrator/Desktop/BlobinatorPro.png', facecolor='w',bbox_inches='tight')