# pip install opencv-python numpy

import cv2
import numpy as np

# Load image in grayscale
image = cv2.imread('target.png', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image 'target.png' not found.")

# --- Saliency Map using Spectral Residual ---
def compute_saliency(gray_img):
    img_float = np.float32(gray_img)
    dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude, angle = cv2.cartToPolar(dft_shift[:,:,0], dft_shift[:,:,1])
    log_magnitude = np.log1p(magnitude)
    spectral_residual = log_magnitude - cv2.boxFilter(log_magnitude, -1, (3,3))
    exp_residual = np.exp(spectral_residual)
    real, imag = cv2.polarToCart(exp_residual, angle)
    dft_shift[:,:,0] = real
    dft_shift[:,:,1] = imag
    dft_ishift = np.fft.ifftshift(dft_shift)
    img_back = cv2.idft(dft_ishift)
    saliency = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    # Normalize to 0-255
    saliency = cv2.GaussianBlur(saliency, (9,9), 2.5)
    saliency = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX)
    saliency = saliency.astype(np.uint8)
    return saliency

saliency_map = compute_saliency(image)
cv2.imwrite('saliency_map.jpg', saliency_map)

# --- Binary Threshold ---
thresh_value = 128
_, binary_map = cv2.threshold(saliency_map, thresh_value, 255, cv2.THRESH_BINARY)
cv2.imwrite('saliency_binary.png', binary_map)
