# Import required libraries
import rasterio
from rasterio import plot
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2  # OpenCV library

# Ensure the working directory exists for output
output_dir = r'D:\CGIP mini project\Landsat\NDVIcalculationLandsat8imageswithPythonandRasterio\Output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List files in the Landsat8 directory to confirm the files exist
# print(os.listdir(r'D:\CGIP mini project\Landsat\NDVIcalculationLandsat8imageswithPythonandRasterio\Landsat8\Set 2'))

# Import bands as separate 1 band raster
band4_path = r'D:\CGIP mini project\Landsat\NDVIcalculationLandsat8imageswithPythonandRasterio\Landsat8\Set 1\Set1_B4.tif'
band5_path = r'D:\CGIP mini project\Landsat\NDVIcalculationLandsat8imageswithPythonandRasterio\Landsat8\Set 1\Set1_B5.tif'
band4 = rasterio.open(band4_path)  # red
band5 = rasterio.open(band5_path)  # nir

# Generate nir and red objects as arrays in float64 format
red = band4.read(1).astype('float64')
nir = band5.read(1).astype('float64')

# NDVI calculation, empty cells or nodata cells are reported as 0
ndvi = np.where((nir + red) == 0., 0, (nir - red) / (nir + red))
# print("NDVI Array (First 5x5):", ndvi[:5, :5])

# # Plot band
# plot.show(band4, title="Band 4 - Red")

# Export NDVI image using rasterio
ndvi_image_path = os.path.join(output_dir, 'ndviImage.tiff')
with rasterio.open(ndvi_image_path, 'w', driver='GTiff', width=band4.width, height=band4.height, count=1, crs=band4.crs, transform=band4.transform, dtype='float64') as ndvi_image:
    ndvi_image.write(ndvi, 1)

# Read NDVI image using OpenCV without altering its values
ndvi_cv2 = cv2.imread(ndvi_image_path, cv2.IMREAD_UNCHANGED)

# Convert NDVI values to CV_8UC1 format for colormap visualization
ndvi_normalized = cv2.normalize(ndvi_cv2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
#cv2.NORM_MINMAX is a normalization types which scales the values such that the minimum values maps to the lower boundary(0) and maximum value maps to the upper boundary(255)
#cv2.CV_8UC1 is an output type in this case grayscale 8 bit image


# Apply colormap (Jet) for visualization
ndvi_colormap = cv2.applyColorMap(ndvi_normalized, cv2.COLORMAP_JET)

# Display NDVI image with colormap overlay using OpenCV
# cv2.imshow('NDVI Image with Colormap', ndvi_colormap)

# Perform edge detection on NDVI image
edges = cv2.Canny(ndvi_normalized, 100, 200)

# Create a mask from edges (convert to 3-channel for compatibility)
edges_mask = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# Blend NDVI image with edges overlay
alpha = 0.5  # Adjust transparency of edges overlay
overlay = cv2.addWeighted(ndvi_colormap, 1 - alpha, edges_mask, alpha, 0)

# Display NDVI image with edges overlay in another window
cv2.imshow('NDVI Image with Edges Overlay', overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the NDVI image with colormap overlay
colormap_output_path = os.path.join(output_dir, 'ndvi_colormap.png')
cv2.imwrite(colormap_output_path, ndvi_colormap)

# Plot NDVI using matplotlib for comparison
with rasterio.open(ndvi_image_path) as ndvi:
    fig = plt.figure(figsize=(18, 12))
    plot.show(ndvi, title="NDVI Image")
