import numpy as np
import matplotlib.pyplot as plt
from time import time
from skimage import io

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')


    ### YOUR CODE HERE
    # Perform convolution
    for i in range(Hi):
        for j in range(Wi):
            # Extract the region of interest from the padded image
            region = padded[i:i+Hk, j:j+Wk]
            # Perform element-wise multiplication with the kernel and sum
            out[i, j] = np.sum(region * kernel)
    ### END YOUR CODE

    return out



def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian filter_values formula,
    and creates a filter_values matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp

    Args:
        size: int of the size of output matrix
        sigma: float of sigma to calculate filter_values

    Returns:
        filter_values: numpy array of shape (size, size)
    """

    filter_values = np.zeros((size, size))
    delta = (size-1) / 2

    ### YOUR CODE HERE
    constant = 1 / (2 * np.pi * sigma**2)

    for i in range(size):
        for j in range(size):
            x = i - delta
            y = j - delta
            filter_values[i, j] = constant * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    ### END YOUR CODE

    return filter_values



def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: x-derivative image
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[-0.5, 0, 0.5]])

    # Compute the convolution of the input image with the kernel
    out = conv(img, kernel)
    ### END YOUR CODE

    return out



def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: y-derivative image
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[-0.5], [0], [0.5]])

    # Compute the convolution of the input image with the kernel
    out = conv(img, kernel)
    ### END YOUR CODE

    return out



def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W)

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W)
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W)
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    dx = partial_x(img)
    dy = partial_y(img)

    # Compute gradient magnitude
    G = np.sqrt(dx**2 + dy**2)

    # Compute gradient direction
    theta = np.arctan2(dy, dx) * (180 / np.pi)
    theta[theta < 0] += 360
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W)
        theta: direction of gradients with shape of (H, W)

    Returns:
        out: non-maxima suppressed image
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    ### BEGIN YOUR CODE
    # Define offsets for 8-connected neighborhood
    offsets_x = [1, 1, 0, -1, -1, -1, 0, 1]
    offsets_y = [0, 1, 1, 1, 0, -1, -1, -1]

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            angle = theta[y, x]
            dx1, dy1 = offsets_x[int((angle % 360) / 45)], offsets_y[int((angle % 360) / 45)]
            dx2, dy2 = -dx1, -dy1

            # Check if the current pixel is a local maximum along the gradient direction
            if (G[y, x] >= G[y + dy1, x + dx1]) and (G[y, x] >= G[y + dy2, x + dx2]):
                out[y, x] = G[y, x]
    ### END YOUR CODE

    return out


def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response
        high: high threshold(float) for strong edges
        low: low threshold(float) for weak edges

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values above
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values below the
            higher threshould and above the lower threshold.
    """

    strong_edges = np.zeros(img.shape)
    weak_edges = np.zeros(img.shape)

    ### YOUR CODE HERE
    # Find strong edges
    strong_edges = img > high

    # Find weak edges
    weak_edges = (img >= low) & (img <= high)
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x)

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel
        H, W: size of the image
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)]
    """
    neighbors = []

    ### YOUR CODE HERE
    # Define offsets for 8-connected neighborhood
    offsets_x = [1, 1, 0, -1, -1, -1, 0, 1]
    offsets_y = [0, 1, 1, 1, 0, -1, -1, -1]

    for i in range(len(offsets_x)):
        neighbor_y = y + offsets_y[i]
        neighbor_x = x + offsets_x[i]
        if 0 <= neighbor_y < H and 0 <= neighbor_x < W and (neighbor_y, neighbor_x) != (y, x):
            neighbors.append((neighbor_y, neighbor_x))
    ### END YOUR CODE

    return neighbors



def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W)
        weak_edges: binary image of shape (H, W)
    Returns:
        edges: numpy array of shape(H, W)
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W))
    ### YOUR CODE HERE
    # Define offsets for 8-connected neighborhood
    offsets_x = [1, 1, 0, -1, -1, -1, 0, 1]
    offsets_y = [0, 1, 1, 1, 0, -1, -1, -1]

    # Iterate over each pixel in strong_edges
    strong_pixels = list(zip(*np.nonzero(strong_edges)))
    visited = np.zeros((H, W), dtype=bool)
    for y, x in strong_pixels:
        # Perform breadth-first search
        queue = [(y, x)]
        visited[y, x] = True
        while queue:
            cy, cx = queue.pop(0)
            edges[cy, cx] = 1
            # Check neighboring pixels in weak_edges
            for offset_y, offset_x in zip(offsets_y, offsets_x):
                ny, nx = cy + offset_y, cx + offset_x
                if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx] and weak_edges[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
    ### END YOUR CODE

    return edges


def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W)
        kernel_size: int of size for kernel matrix
        sigma: float for calculating kernel
        high: high threshold for strong edges
        low: low threashold for weak edges
    Returns:
        edge: numpy array of shape(H, W)
    """
    ### YOUR CODE HERE
    # Step 1: Apply Gaussian smoothing
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed = conv(img, kernel)
    # Step 2: Compute gradient magnitude and direction
    G, theta = gradient(smoothed)

    # Step 3: Perform non-maximum suppression
    nms = non_maximum_suppression(G, theta)

    # Step 4: Apply double thresholding
    strong_edges, weak_edges = double_thresholding(nms, high, low)

    # Step 5: Link weak edges to strong edges
    edge = link_edges(strong_edges, weak_edges)
    ### END YOUR CODE

    return edge