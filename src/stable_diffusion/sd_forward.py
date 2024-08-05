import numpy as np
import cv2
from PIL import Image
from diffusers.utils import make_image_grid
from matplotlib import pyplot as plt

# load image
image = cv2.imread('../../images/maomi.jpg')

# denoise steps
num_iterations = 16
# beta
beta = 0.1

# save all noised images
images = []

image = image.astype(np.float64)
# iteration
for i in range(num_iterations):
    # mean
    mean = np.sqrt(1 - beta) * image
    # sample based on x0
    image += np.random.normal(mean, beta, image.shape)
    # for show
    pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
    images.append(pil_image)

# show image with grid
show_image = make_image_grid(images, rows=4, cols=4)
show_image.show()

sampled_image = image
print(sampled_image.shape)
plt.scatter(sampled_image[:, 0], sampled_image[:, 1])
plt.title('Sampled image')
plt.show()