import numpy as np
import cv2
from PIL import Image
from diffusers.utils import make_image_grid
from matplotlib import pyplot as plt

# load image
image = cv2.imread('../../images/maomi.jpg')

# denoise steps
steps = 16
# beta
beta = 0.1

# save all noised images
images = []

image = image / 255.0
# iteration
for i in range(steps):
    # mean
    mean = np.sqrt(1 - beta) * image
    # sample
    image = np.random.normal(mean, beta, image.shape)
    # for show
    show = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(show, 'RGB')
    images.append(pil_image)

# show image with grid
show_image = make_image_grid(images, rows=4, cols=4)
show_image.show('../../outputs/maomi.jpg')

sampled_image = image
plt.scatter(sampled_image[:, 0], sampled_image[:, 1])
plt.title('Sampled image')
plt.show()