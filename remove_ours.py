# basics
import numpy as np
np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt

# customs
from src.estimate_watermark import *
from src.preprocess import *
from src.image_crawler import *
from src.watermark_reconstruct import *


gx, gy, gxlist, gylist = estimate_watermark('sources/paired_train-wmark_cifar10_wmark_32x32_tile_cifar10_transpair_noise_ResNet_l1_binary-cross-entropy_cross-entropy_64_30_0.01_0.001_0.001_0_0_2.0_1.0_1.0_10.0_1.0')

# est = poisson_reconstruct(gx, gy, np.zeros(gx.shape)[:,:,0])
cropped_gx, cropped_gy = crop_watermark(gx, gy)
W_m = poisson_reconstruct(cropped_gx, cropped_gy)

# random photo
img = cv2.imread(
	('sources/paired_train-wmark_cifar10_wmark_32x32_tile_cifar10_'
	 'transpair_noise_ResNet_l1_binary-cross-entropy_cross-entropy_'
	 '64_30_0.01_0.001_0.001_0_0_2.0_1.0_1.0_10.0_1.0/0_4.png'))
im, start, end = watermark_detector(img, cropped_gx, cropped_gy)

# plot the watermarked area
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
plt.show()

# we don't need the others
exit()

# We are done with watermark estimation
# W_m is the cropped watermark
num_images = len(gxlist)

J, img_paths = get_cropped_images('images/fotolia_processed', num_images, start, end, cropped_gx.shape)
# get a random subset of J
idx = [389, 144, 147, 468, 423, 92, 3, 354, 196, 53, 470, 445, 314, 349, 105, 366, 56, 168, 351, 15, 465, 368, 90, 96, 202, 54, 295, 137, 17, 79, 214, 413, 454, 305, 187, 4, 458, 330, 290, 73, 220, 118, 125, 180, 247, 243, 257, 194, 117, 320, 104, 252, 87, 95, 228, 324, 271, 398, 334, 148, 425, 190, 78, 151, 34, 310, 122, 376, 102, 260]
idx = idx[:25]
# Wm = (255*PlotImage(W_m))
Wm = W_m - W_m.min()

# get threshold of W_m for alpha matte estimate
alph_est = estimate_normalized_alpha(J, Wm)
alph = np.stack([alph_est, alph_est, alph_est], axis=2)
C, est_Ik = estimate_blend_factor(J, Wm, alph)

alpha = alph.copy()
for i in xrange(3):
	alpha[:,:,i] = C[i]*alpha[:,:,i]

Wm = Wm + alpha*est_Ik

W = Wm.copy()
for i in xrange(3):
	W[:,:,i]/=C[i]

Jt = J[:25]
# now we have the values of alpha, Wm, J
# Solve for all images
Wk, Ik, W, alpha1 = solve_images(Jt, W_m, alpha, W)
# W_m_threshold = (255*PlotImage(np.average(W_m, axis=2))).astype(np.uint8)
# ret, thr = cv2.threshold(W_m_threshold, 127, 255, cv2.THRESH_BINARY)
