import numpy as np
import cv2
from scipy.ndimage import convolve


def compute_gradient_energy(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    filter_dv = np.stack([filter_dv] * 3, axis=2)

    img = img.astype('float32')
    convolved = np.abs(convolve(img, filter_du)) + np.abs(convolve(img, filter_dv))
    gradient_energy = convolved.sum(axis=2)
    # denoise the gradient energy
    gradient_energy = cv2.GaussianBlur(gradient_energy, (5, 5), 0.5)
    return gradient_energy



#find fowrwad energy
def compute_forward_energy(img):
    r, c = img.shape
    energy = np.zeros((r, c))
    m, n = energy.shape
    for i in range(1, m):
        for j in range(n):
            if j == 0:
                energy[i, j] = img[i, j] + min(energy[i - 1, j], energy[i - 1, j + 1])
            elif j == n - 1:
                energy[i, j] = img[i, j] + min(energy[i - 1, j - 1], energy[i - 1, j])
            else:
                energy[i, j] = img[i, j] + min(energy[i - 1, j - 1], energy[i - 1, j], energy[i - 1, j + 1])
    return energy


def normalize_image(img):
    img = img.astype('float32')
    normalized = (img - img.min()) / (img.max() - img.min())
    return normalized

def compute_combined_energy(img, depth_map,saliency_map, alpha=0.5, beta=0, gamma=0.5):
    gradient_energy = normalize_image(compute_gradient_energy(img))
    # cv2.imshow('image', gradient_energy)
    # cv2.waitKey()
    saliency_energy = normalize_image(saliency_map)
    depth_energy = normalize_image(depth_map)

    combined_energy = (alpha * gradient_energy +
                       beta * (saliency_energy) +
                       gamma * (depth_energy))
    return combined_energy


def find_seam(img,energy_map):
    r, c = energy_map.shape
    cost = np.zeros((r, c))
    cost[0, :] = energy_map[0, :]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = normalize_image(img)
    for i in range(1, r):
        for j in range(c):
            if j == 0:
                min_cost = min(cost[i - 1, j], cost[i - 1, j + 1] + 2 * abs(img[i, j + 1] - img[i - 1, j]))
            elif j == c - 1:
                min_cost = min(cost[i - 1, j - 1] + 2 * abs(img[i, j - 1] - img[i - 1, j]), cost[i - 1, j])
            else:
                min_cost = min(cost[i - 1, j - 1] + abs(img[i, j - 1] - img[i - 1, j]) + abs(img[i][j - 1] - img[i][j + 1]), cost[i - 1, j] + abs(img[i][j - 1] - img[i][j + 1]), cost[i - 1, j + 1] + abs(img[i, j + 1] - img[i - 1, j]) + abs(img[i][j - 1] - img[i][j + 1]))
            cost[i, j] = energy_map[i, j] + min_cost

    seam = np.zeros(r, dtype=np.int32)
    seam[-1] = np.argmin(cost[-1])
    for i in range(r - 2, -1, -1):
        prev_x = seam[i + 1]
        if prev_x == 0:
            seam[i] = np.argmin(cost[i, :2])
        else:
            seam[i] = prev_x + np.argmin(cost[i, prev_x - 1:prev_x + 2]) - 1
    return seam



def remove_seam(img, seam):
    r, c, _ = img.shape
    output = np.zeros((r, c - 1, 3), dtype=np.uint8)
    for i in range(r):
        output[i, :, 0] = np.delete(img[i, :, 0], seam[i])
        output[i, :, 1] = np.delete(img[i, :, 1], seam[i])
        output[i, :, 2] = np.delete(img[i, :, 2], seam[i])

    for i in range(r):
        img[i, seam[i], :] = [0, 0, 255]

    # cv2.imshow('image', img)
    # cv2.waitKey()
    return output


def remove_seam2(img, seam):
    r, c = img.shape
    output = np.zeros((r, c - 1), dtype=np.uint8)
    for i in range(r):
        output[i, :] = np.delete(img[i, :], seam[i])
    return output

def seam_carve(img, depth_map, saliency_map, num_seams):
    for _ in range(num_seams):
        energy_map = compute_combined_energy(img, depth_map, saliency_map)
        seam = find_seam(img, energy_map)
        img = remove_seam(img, seam)
        depth_map = remove_seam2(depth_map, seam)
        #show seams on the image
        saliency_map = remove_seam2(saliency_map, seam)
    return img


# Read the input image and depth map
input_img = cv2.imread('Diana\Diana.png')
depth_map = cv2.imread('Diana\Diana_DMap.png', cv2.IMREAD_GRAYSCALE)
saliency_map =cv2.imread('Diana\Diana_SMap.png', cv2.IMREAD_GRAYSCALE)
# Number of seams to remove
num_seams = 150

# Perform seam carving
output_img = seam_carve(input_img, depth_map,saliency_map, num_seams)

# Save the result
cv2.imwrite('Diana505.png', output_img)
