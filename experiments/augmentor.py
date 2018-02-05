import numpy as np
import scipy as sp


def resize(arr, size):
    """Resizes an image to ``size``.

    Args:
        arr (2D array): image.

    Returns:
        resize (2D array): resized image.
    """
    if (size == 1.0) or (size == 100):
        return arr

    return sp.misc.imresize(arr, size, interp='nearest')


def rotate(arr, degree):
    """Rotates an image by ``degree``.

    Args:
        arr (2D array): image.

    Returns:
        rotate (2D array): rotated image.
    """
    if degree % 360 == 0:
        return arr

    return sp.ndimage.rotate(arr, degree, reshape=False, order=0)


def batch_process(arrs, ops=None):
    """Processes batch of images.

    Args:
        arrs (3D array): images. 2nd and 3rd axis span images.
        ops (list of tuples): list of (operation, argument). operation can be
                              'resize' and 'rotate'.
    Returns:
        (3D array): processed images.
    """
    processed_arrs = []
    for arr in arrs:
        for op in ops:
            if op[0] == 'resize':
                arr = resize(arr, op[1])
            if op[0] == 'rotate':
                arr = rotate(arr, op[1])
        processed_arrs.append(arr)

    return np.array(processed_arrs)
