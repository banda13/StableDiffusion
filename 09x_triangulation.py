import math

import cv2
import numpy as np
from PIL import Image
from skimage.io import imsave
from tqdm import tqdm
from scipy.spatial import Delaunay
from skimage.draw import polygon
from skimage.feature import canny
from skimage.transform import pyramid_reduce

reduce = 0.5 # 0-1: How much % reduce the number of triangles
coloring_mode = "CENTER"  # MEAN

""" Read img"""
img = cv2.imread('output/00004-2377551714.png')
assert img is not None, "file could not be read, check with os.path.exists()"

""" Optional resize and convert to gray """
img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

""" Detect edges and reduce the number of them"""
edges = canny(img_gray, sigma=3, low_threshold=0.05, high_threshold=0.1)
candidates = np.array([idx for idx, weight in np.ndenumerate(edges) if weight >= 0.2])
sample_points = candidates[np.random.choice(candidates.shape[0], size=int(candidates.shape[0] * reduce), replace=False)]

""" Create Delaunay triangualation """
triangulated: Delaunay = Delaunay(sample_points)
triangles = sample_points[triangulated.simplices]
print(f"{len(triangles)} triangle detected")

""" Color the triangles """
res = np.empty(shape=(2 * img.shape[0], 2 * img.shape[1], img.shape[2]), dtype=np.uint8)
for triangle in tqdm(triangles):
    if coloring_mode == "CENTER":
        i, j = polygon(2 * triangle[:, 0], 2 * triangle[:, 1], res.shape)
        res[i, j] = img[tuple(np.mean(triangle, axis=0, dtype=np.int32))]
    else:
        i, j = polygon(2 * triangle[:, 0], 2 * triangle[:, 1], res.shape)
        res[i, j] = np.mean(img[polygon(triangle[:, 0], triangle[:, 1], img.shape)], axis=0)

""" Reduce the number of pyramids """
res = pyramid_reduce(res, channel_axis=2)
cv2.imwrite("output/yape4x.jpg", (res*255))
# cv2.imshow('Result', res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()