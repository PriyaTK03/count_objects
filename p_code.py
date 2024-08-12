import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# Define file paths
input_image_path =  r"C:\Users\priya\Documents\count_objects\Ref\image.jpeg"
grayscale_image_path = r"C:\Users\priya\Documents\Count number of objects\preprocessed\grayscale_image.jpg"
edges_image_path = r"C:\Users\priya\Documents\Count number of objects\preprocessed\edges.jpg"
contour_image_path = r"C:\Users\priya\Documents\Count number of objects\preprocessed\contour_image.jpg"
plot_image_path = r"C:\Users\priya\Documents\Count number of objects\preprocessed\plot_with_edge_and_contour.png"

# Create the output directory if it doesn't exist
output_dir = os.path.dirname(grayscale_image_path)
os.makedirs(output_dir, exist_ok=True)
image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)

# Convert the image to grayscale
gray_scaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Saving the grayscaled image
cv2.imwrite(grayscale_image_path, gray_scaled)

# Blur the image
blurred = cv2.GaussianBlur(gray_scaled, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Saving the edge image
cv2.imwrite(edges_image_path, edges)

# Dilating the edges
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=1)

# Finding contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtering contours based on area
min_area = 100
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# Draw contours on a blank image
contour_image = np.zeros_like(image)
cv2.drawContours(contour_image, filtered_contours, -1, (255, 255, 255), 2)

# Save the image with contours
cv2.imwrite(contour_image_path, contour_image)

# Count the number of contours 
number_of_objects = len(filtered_contours)

print(f"Number of objects counted: {number_of_objects}")

# Plot the original, edge, and contour images
plt.figure(figsize=(10, 5))

plt.subplot(131)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.xticks([]), plt.yticks([])

plt.subplot(132)
plt.imshow(edges, cmap='gray')
plt.title('Edge Image')
plt.xticks([]), plt.yticks([])

plt.subplot(133)
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.title('Image with Contours')
plt.xticks([]), plt.yticks([])

# Save the plot as an image file
plt.savefig(plot_image_path)

plt.tight_layout()
plt.show()
