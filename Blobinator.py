import skimage.io
import skimage.feature
import skimage.morphology
import skimage.filters
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import cv2
import imagecodecs
import numpy as np
from skimage.morphology import binary_closing, disk
import scipy.ndimage as nd


filepath='F:/EVOS/F.d6.500-2.0030.tif'




image = cv2.imread(filepath) 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize=(8, 8)) 
claheNorm = clahe.apply(gray) 
cv2.imwrite(filepath+'claheNorm.tif', claheNorm) 




fig=plt.figure(figsize=(50, 20))




def plotRoi(spots, img_ax, color, radius):
    patches = []
    for spot in spots:
        y, x , z= spot
        c = plt.Circle((x, y), radius)
        patches.append(c)
    img_ax.add_collection(PatchCollection(patches, facecolors = "None", edgecolors = color, alpha = 1, linewidths = 3))


img = skimage.io.imread(filepath+'claheNorm.tif')
img = img[:,:]


ax= fig.add_subplot(1,3,1)

ax.imshow(image, cmap = "Greys")
plt.title('Original',size=30)
#img= skimage.filters.gaussian(img, sigma=8)

ax= fig.add_subplot(1,3,2)

ax.imshow(img, cmap = "Greys")
#spots = skimage.feature.blob_dog(img, min_sigma=1, max_sigma=50, sigma_ratio=12, threshold=0.95, overlap=1, exclude_border=False)
spots = skimage.feature.blob_log(img, min_sigma=0.395, max_sigma=50, num_sigma=10, threshold=0.95, overlap=0.99, log_scale=False,exclude_border=False)

plotRoi(spots, ax, "red", radius = 10)
plt.title('Dead Cells')








image = skimage.io.imread(filepath+'claheNorm.tif')
edges = skimage.filters.sobel(image)

low = 0.03
high = 0.35

lowt = (edges > low).astype(int)
hight = (edges > high).astype(int)
hyst = skimage.filters.apply_hysteresis_threshold(edges, low, high)

ax.imshow(lowt,cmap='viridis')
plt.title('Blobinator Classification',size=30)



strel = disk(4)
I_closed = binary_closing(lowt, strel)
I_closed_filled = nd.morphology.binary_fill_holes(I_closed)
cell_mass= cv2.countNonZero(np.float32(I_closed_filled))







#allcells=len(spots2)
deadcells=len(spots)

print("Ratio viable to dead:" + str(cell_mass/deadcells))

#fig.savefig('C:/Users/Administrator/Desktop/Blobinator.png', facecolor='w',bbox_inches='tight')