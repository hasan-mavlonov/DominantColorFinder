import matplotlib
matplotlib.use('TkAgg')  # Use a different backend compatible with Matplotlib
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Load the image
image_path = "IMG_2243.JPG"
image = Image.open(image_path)

# Resize image for easier processing
image_resized = image.resize((100, 100))
image_array = np.array(image_resized)
image_array = image_array.reshape(-1, 3)

# Use KMeans to find the dominant colors
kmeans = KMeans(n_clusters=3, random_state=0).fit(image_array)
dominant_colors = kmeans.cluster_centers_

# Display the dominant colors as a palette
fig, ax = plt.subplots(1, len(dominant_colors), figsize=(12, 4))
for i, color in enumerate(dominant_colors):
    ax[i].imshow([[color / 255]])
    ax[i].axis("off")
plt.tight_layout()
plt.show()

# Return the RGB values of dominant colors
dominant_colors
