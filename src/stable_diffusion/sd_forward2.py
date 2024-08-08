import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from itertools import accumulate


image = cv2.imread('../../images/maomi.jpg')
image = image / 255.0  # [0, 1]
image = image * 2 - 1  # [-1, 1]

# denoise steps
steps = 1000
# beta
beta_s = 0.001
beta_e = 0.02
betas = np.linspace(beta_s, beta_e, steps)

alphas = [1 - beta for beta in betas]
alpha_bars = list(accumulate(alphas, lambda a, b: a * b))

interested_step = 200
image_in_step_10 = (np.sqrt(alpha_bars[interested_step]) * image +
                    np.sqrt(1 - alpha_bars[interested_step]) * np.random.normal(0, 1, image.shape))

image_in_step_10 = (image_in_step_10 + 1) / 2  # [0, 1]

cv_img = cv2.cvtColor((image_in_step_10 * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
out = Image.fromarray(cv_img, "RGB")
out.show()
