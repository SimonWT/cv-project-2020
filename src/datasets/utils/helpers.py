import torch
import numpy as np


def generate_point_heatmap(heatmap_shape, pt, sigma, label_type='Gaussian'):
    """
    img: image you want the shape of heatmap to match to
    pt: point of shape (2)
    return: ndarray
    """
    img = np.zeros(heatmap_shape)

    tmp_size = sigma * 3
    wrap_float = lambda x: int(np.around(x))
    ul = [wrap_float(pt[0] - tmp_size), wrap_float(pt[1] - tmp_size)]
    br = [wrap_float(pt[0] + tmp_size + 1), wrap_float(pt[1] + tmp_size + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img[..., 0]

    # Generate gaussian
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if label_type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    else:
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])
    if len(img.shape) == 3:
        img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1], np.newaxis]
        img = img[..., 0]
    elif len(img.shape) == 2:
        img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    else:
        raise NotImplementedError("Only 2 & 3 channel images are supported")
    return img


def generate_target_heatmap(heatmap_shape, target, sigma):
    """
    target: points array of shape (num_of_points, 2)
    """
    maps = []
    for point in target:
        maps.append(generate_point_heatmap(heatmap_shape, point, sigma))
    ret = np.stack(maps, axis=-1)
    return ret
