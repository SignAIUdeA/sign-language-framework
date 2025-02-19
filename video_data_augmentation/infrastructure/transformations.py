from data_augmentation.domain.entities import VideoData
from PIL import ImageOps
import scipy
import skimage.transform
import PIL
import numpy as np
import random
import cv2
import numbers

def apply_mirror(video: VideoData, parameters: dict) -> VideoData:
    """
    Aplica una transformación de espejo al video.

    Parámetros:
    - video (VideoData): El video al que se le aplicará la transformación.
    - parameters (dict): Diccionario con los parámetros de la transformación.
        - No se requieren parámetros adicionales para esta transformación.

    Ejemplo de uso:
    parameters = {}
    """
    if isinstance(video[0], np.ndarray):
        return [np.fliplr(img) for img in video]
    elif isinstance(video[0], PIL.Image.Image):
        return [img.transpose(PIL.Image.FLIP_LEFT_RIGHT) for img in video]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image but got list of {0}'.format(type(video[0])))



def apply_piecewise_affine_transform(video: VideoData, parameters: dict) -> VideoData:
    """
    Aplica una transformación afín por partes al video.

    Parámetros:
    - video (VideoData): El video al que se le aplicará la transformación.
    - parameters (dict): Diccionario con los parámetros de la transformación.
        - 'displacement' (int): Desplazamiento para la transformación.
        - 'displacement_kernel' (int): Kernel de desplazamiento para el efecto borroso.
        - 'displacement_magnification' (float): Magnificación del desplazamiento.

    Ejemplo de uso:
    parameters = {
        'displacement': 5,
        'displacement_kernel': 3,
        'displacement_magnification': 1.5
    }
    """
    displacement = parameters.get('displacement', 0)
    displacement_kernel = parameters.get('displacement_kernel', 0)
    displacement_magnification = parameters.get('displacement_magnification', 0)

    ret_img_group = video
    if isinstance(video[0], np.ndarray):
        im_size = video[0].shape
        image_w, image_h = im_size[1], im_size[0]
    elif isinstance(video[0], PIL.Image.Image):
        im_size = video[0].size
        image_w, image_h = im_size[0], im_size[1]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image but got list of {0}'.format(type(video[0])))

    displacement_map = np.random.rand(image_h, image_w, 2) * 2 * displacement - displacement
    displacement_map = cv2.GaussianBlur(displacement_map, None, displacement_kernel)
    displacement_map *= displacement_magnification * displacement_kernel
    displacement_map = np.floor(displacement_map).astype('int32')

    displacement_map_rows = displacement_map[..., 0] + np.tile(np.arange(image_h), (image_w, 1)).T.astype('int32')
    displacement_map_rows = np.clip(displacement_map_rows, 0, image_h - 1)

    displacement_map_cols = displacement_map[..., 1] + np.tile(np.arange(image_w), (image_h, 1)).astype('int32')
    displacement_map_cols = np.clip(displacement_map_cols, 0, image_w - 1)

    if isinstance(video[0], np.ndarray):
        return [img[(displacement_map_rows.flatten(), displacement_map_cols.flatten())].reshape(img.shape) for img in video]
    elif isinstance(video[0], PIL.Image.Image):
        return [PIL.Image.fromarray(np.asarray(img)[(displacement_map_rows.flatten(), displacement_map_cols.flatten())].reshape(np.asarray(img).shape)) for img in video]



def apply_superpixel(video: VideoData, parameters: dict) -> VideoData:
    """
    Aplica una transformación de superpíxeles al video.

    Parámetros:
    - video (VideoData): El video al que se le aplicará la transformación.
    - parameters (dict): Diccionario con los parámetros de la transformación.
        - 'p_replace' (float): Probabilidad de que un área de superpíxeles sea reemplazada.
        - 'n_segments' (int): Número de superpíxeles a generar.

    Ejemplo de uso:
    parameters = {
        'p_replace': 0.5,
        'n_segments': 100
    }
    """
    p_replace = parameters.get('p_replace', 0.5)
    n_segments = parameters.get('n_segments', 100)

    is_PIL = isinstance(video[0], PIL.Image.Image)
    if is_PIL:
        video = [np.asarray(img) for img in video]

    replace_samples = np.tile(np.array([p_replace]), n_segments)
    avg_image = np.mean(video, axis=0)
    segments = skimage.segmentation.slic(avg_image, n_segments=n_segments, compactness=10)

    if not np.max(replace_samples) == 0:
        video = [_apply_segmentation(img, replace_samples, segments) for img in video]

    if is_PIL:
        return [PIL.Image.fromarray(img) for img in video]
    else:
        return video

def _apply_segmentation(image, replace_samples, segments):
    nb_channels = image.shape[2]
    image_sp = np.copy(image)
    for c in range(nb_channels):
        regions = skimage.measure.regionprops(segments + 1, intensity_image=image[..., c])
        for ridx, region in enumerate(regions):
            if replace_samples[ridx % len(replace_samples)] == 1:
                mean_intensity = region.mean_intensity
                image_sp_c = image_sp[..., c]
                image_sp_c[segments == ridx] = mean_intensity

    return image_sp



def apply_gaussian_blur(video: VideoData, parameters: dict) -> VideoData:
    """
    Aplica un desenfoque gaussiano al video.

    Parámetros:
    - video (VideoData): El video al que se le aplicará la transformación.
    - parameters (dict): Diccionario con los parámetros de la transformación.
        - 'sigma' (float): Desviación estándar para el desenfoque gaussiano.

    Ejemplo de uso:
    parameters = {
        'sigma': 2.0
    }
    """
    sigma = parameters.get('sigma', 2.0)

    if isinstance(video[0], np.ndarray):
        return [scipy.ndimage.gaussian_filter(img, sigma=sigma, order=0) for img in video]
    elif isinstance(video[0], PIL.Image.Image):
        return [img.filter(PIL.ImageFilter.GaussianBlur(radius=sigma)) for img in video]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image but got list of {0}'.format(type(video[0])))




def apply_invert_color(video: VideoData, parameters: dict) -> VideoData:
    """
    Invierte los colores del video.

    Parámetros:
    - video (VideoData): El video al que se le aplicará la transformación.
    - parameters (dict): Diccionario con los parámetros de la transformación.
        - No se requieren parámetros adicionales para esta transformación.

    Ejemplo de uso:
    parameters = {}
    """
    if isinstance(video[0], np.ndarray):
        return [np.invert(img) for img in video]
    elif isinstance(video[0], PIL.Image.Image):
        return [ImageOps.invert(img) for img in video]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image but got list of {0}'.format(type(video[0])))


def apply_random_rotate(video: VideoData, parameters: dict) -> VideoData:
    """
    Aplica una rotación aleatoria al video.

    Parámetros:
    - video (VideoData): El video al que se le aplicará la transformación.
    - parameters (dict): Diccionario con los parámetros de la transformación.
        - 'degrees' (tuple or int): Rango de grados para seleccionar aleatoriamente. Si es un número, el rango será (-degrees, +degrees).

    Ejemplo de uso:
    parameters = {
        'degrees': 30
    }
    """
    degrees = parameters.get('degrees', 30)
    if isinstance(degrees, numbers.Number):
        degrees = (-degrees, degrees)
    
    angle = random.uniform(degrees[0], degrees[1])
    if isinstance(video[0], np.ndarray):
        rotated = [skimage.transform.rotate(img, angle) for img in video]
    elif isinstance(video[0], PIL.Image.Image):
        rotated = [img.rotate(angle) for img in video]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image but got list of {0}'.format(type(video[0])))

    return rotated



def apply_random_resize(video: VideoData, parameters: dict) -> VideoData:
    """
    Aplica un cambio de tamaño aleatorio al video.

    Parámetros:
    - video (VideoData): El video al que se le aplicará la transformación.
    - parameters (dict): Diccionario con los parámetros de la transformación.
        - 'rate' (float): El video se escala uniformemente entre [1 - rate, 1 + rate].
        - 'interp' (str): Interpolación a usar para el cambio de tamaño ('nearest', 'lanczos', 'bilinear', 'bicubic' o 'cubic').

    Ejemplo de uso:
    parameters = {
        'rate': 0.2,
        'interp': 'bilinear'
    }
    """
    rate = parameters.get('rate', 0.2)
    interp = parameters.get('interp', 'bilinear')

    scaling_factor = random.uniform(1 - rate, 1 + rate)

    if isinstance(video[0], np.ndarray):
        im_h, im_w, im_c = video[0].shape
    elif isinstance(video[0], PIL.Image.Image):
        im_w, im_h = video[0].size

    new_w = int(im_w * scaling_factor)
    new_h = int(im_h * scaling_factor)
    new_size = (new_h, new_w)
    if isinstance(video[0], np.ndarray):
        return [scipy.misc.imresize(img, size=(new_h, new_w), interp=interp) for img in video]
    elif isinstance(video[0], PIL.Image.Image):
        return [img.resize(size=(new_w, new_h), resample=_get_PIL_interp(interp)) for img in video]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image but got list of {0}'.format(type(video[0])))

def _get_PIL_interp(interp):
    if interp == 'nearest':
        return PIL.Image.NEAREST
    elif interp == 'lanczos':
        return PIL.Image.LANCZOS
    elif interp == 'bilinear':
        return PIL.Image.BILINEAR
    elif interp == 'bicubic':
        return PIL.Image.BICUBIC
    elif interp == 'cubic':
        return PIL.Image.CUBIC



def apply_translate(video: VideoData, parameters: dict) -> VideoData:
    """
    Aplica una traslación al video.

    Parámetros:
    - video (VideoData): El video al que se le aplicará la transformación.
    - parameters (dict): Diccionario con los parámetros de la transformación.
        - 'x' (int): Traslación en el eje X.
        - 'y' (int): Traslación en el eje Y.

    Ejemplo de uso:
    parameters = {
        'x': 10,
        'y': 10
    }
    """
    x = parameters.get('x', 0)
    y = parameters.get('y', 0)

    x_move = random.randint(-x, +x)
    y_move = random.randint(-y, +y)

    if isinstance(video[0], np.ndarray):
        rows, cols, ch = video[0].shape
        transform_mat = np.float32([[1, 0, x_move], [0, 1, y_move]])
        return [cv2.warpAffine(img, transform_mat, (cols, rows)) for img in video]
    elif isinstance(video[0], PIL.Image.Image):
        return [img.transform(img.size, PIL.Image.AFFINE, (1, 0, x_move, 0, 1, y_move)) for img in video]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image but got list of {0}'.format(type(video[0])))



def apply_center_crop(video: VideoData, parameters: dict) -> VideoData:
    """
    Aplica un recorte centrado al video.

    Parámetros:
    - video (VideoData): El video al que se le aplicará la transformación.
    - parameters (dict): Diccionario con los parámetros de la transformación.
        - 'width' (int): Ancho del recorte.
        - 'height' (int): Altura del recorte.

    Ejemplo de uso:
    parameters = {
        'width': 100,
        'height': 100
    }
    """
    crop_h = parameters.get('height', 100)
    crop_w = parameters.get('width', 100)

    if isinstance(video[0], np.ndarray):
        im_h, im_w, im_c = video[0].shape
    elif isinstance(video[0], PIL.Image.Image):
        im_w, im_h = video[0].size
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image but got list of {0}'.format(type(video[0])))

    if crop_w > im_w or crop_h > im_h:
        error_msg = ('Initial image size should be larger than cropped size but got cropped sizes : ' +
                     '({w}, {h}) while initial image is ({im_w}, {im_h})'.format(im_w=im_w, im_h=im_h, w=crop_w, h=crop_h))
        raise ValueError(error_msg)

    w1 = int(round((im_w - crop_w) / 2.))
    h1 = int(round((im_h - crop_h) / 2.))

    if isinstance(video[0], np.ndarray):
        return [img[h1:h1 + crop_h, w1:w1 + crop_w, :] for img in video]
    elif isinstance(video[0], PIL.Image.Image):
        return [img.crop((w1, h1, w1 + crop_w, h1 + crop_h)) for img in video]



def apply_horizontal_flip(video: VideoData, parameters: dict) -> VideoData:
    """
    Aplica un volteo horizontal al video.

    Parámetros:
    - video (VideoData): El video al que se le aplicará la transformación.
    - parameters (dict): Diccionario con los parámetros de la transformación.
        - No se requieren parámetros adicionales para esta transformación.

    Ejemplo de uso:
    parameters = {}
    """
    if isinstance(video[0], np.ndarray):
        return [np.fliplr(img) for img in video]
    elif isinstance(video[0], PIL.Image.Image):
        return [img.transpose(PIL.Image.FLIP_LEFT_RIGHT) for img in video]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image but got list of {0}'.format(type(video[0])))



def apply_vertical_flip(video: VideoData, parameters: dict) -> VideoData:
    """
    Aplica un volteo vertical al video.

    Parámetros:
    - video (VideoData): El video al que se le aplicará la transformación.
    - parameters (dict): Diccionario con los parámetros de la transformación.
        - No se requieren parámetros adicionales para esta transformación.

    Ejemplo de uso:
    parameters = {}
    """
    if isinstance(video[0], np.ndarray):
        return [np.flipud(img) for img in video]
    elif isinstance(video[0], PIL.Image.Image):
        return [img.transpose(PIL.Image.FLIP_TOP_BOTTOM) for img in video]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image but got list of {0}'.format(type(video[0])))



def apply_add(video: VideoData, parameters: dict) -> VideoData:
    """
    Aplica una adición de valor a los píxeles del video.

    Parámetros:
    - video (VideoData): El video al que se le aplicará la transformación.
    - parameters (dict): Diccionario con los parámetros de la transformación.
        - 'value' (int): Valor a añadir a los píxeles.

    Ejemplo de uso:
    parameters = {
        'value': 10
    }
    """
    value = parameters.get('value', 10)

    is_PIL = isinstance(video[0], PIL.Image.Image)
    if is_PIL:
        video = [np.asarray(img) for img in video]

    data_final = []
    for img in video:
        image = img.astype(np.int32)
        image += value
        image = np.where(image > 255, 255, image)
        image = np.where(image < 0, 0, image)
        data_final.append(image.astype(np.uint8))

    if is_PIL:
        return [PIL.Image.fromarray(img) for img in data_final]
    else:
        return data_final



def apply_multiply(video: VideoData, parameters: dict) -> VideoData:
    """
    Aplica una multiplicación de valor a los píxeles del video.

    Parámetros:
    - video (VideoData): El video al que se le aplicará la transformación.
    - parameters (dict): Diccionario con los parámetros de la transformación.
        - 'value' (float): Valor por el cual multiplicar los píxeles.

    Ejemplo de uso:
    parameters = {
        'value': 1.1
    }
    """
    value = parameters.get('value', 1.1)

    is_PIL = isinstance(video[0], PIL.Image.Image)
    if is_PIL:
        video = [np.asarray(img) for img in video]

    data_final = []
    for img in video:
        image = img.astype(np.float64)
        image *= value
        image = np.where(image > 255, 255, image)
        image = np.where(image < 0, 0, image)
        data_final.append(image.astype(np.uint8))

    if is_PIL:
        return [PIL.Image.fromarray(img) for img in data_final]
    else:
        return data_final



def apply_downsample(video: VideoData, parameters: dict) -> VideoData:
    """
    Aplica una reducción de resolución al video.

    Parámetros:
    - video (VideoData): El video al que se le aplicará la transformación.
    - parameters (dict): Diccionario con los parámetros de la transformación.
        - 'factor' (int): Factor de reducción de resolución.

    Ejemplo de uso:
    parameters = {
        'factor': 2
    }
    """
    factor = parameters.get('factor', 2)

    nb_return_frame = np.floor(len(video) / factor).astype(int)
    return_ind = [int(i) for i in np.linspace(1, len(video), num=nb_return_frame)]

    return [video[i-1] for i in return_ind]



def apply_upsample(video: VideoData, parameters: dict) -> VideoData:
    """
    Aplica un aumento de resolución al video.

    Parámetros:
    - video (VideoData): El video al que se le aplicará la transformación.
    - parameters (dict): Diccionario con los parámetros de la transformación.
        - 'factor' (int): Factor de aumento de resolución.

    Ejemplo de uso:
    parameters = {
        'factor': 2
    }
    """
    factor = parameters.get('factor', 2)

    nb_return_frame = np.floor(len(video) * factor).astype(int)
    return_ind = [int(i) for i in np.linspace(1, len(video), num=nb_return_frame)]

    return [video[i-1] for i in return_ind]



def apply_elastic_transformation(video: VideoData, parameters: dict) -> VideoData:
    """
    Aplica una transformación elástica al video.

    Parámetros:
    - video (VideoData): El video al que se le aplicará la transformación.
    - parameters (dict): Diccionario con los parámetros de la transformación.
        - 'alpha' (float): Intensidad de la transformación.
        - 'sigma' (float): Suavidad de la transformación.

    Ejemplo de uso:
    parameters = {
        'alpha': 1.0,
        'sigma': 0.5
    }
    """
    alpha = parameters.get('alpha', 1.0)
    sigma = parameters.get('sigma', 0.5)

    is_PIL = isinstance(video[0], PIL.Image.Image)
    if is_PIL:
        video = [np.asarray(img) for img in video]

    result = []
    for img in video:
        image_first_channel = np.squeeze(img[..., 0])
        indices_x, indices_y = _generate_indices(image_first_channel.shape, alpha=alpha, sigma=sigma)
        result.append(_map_coordinates(img, indices_x, indices_y))

    if is_PIL:
        return [PIL.Image.fromarray(img) for img in result]
    else:
        return result

def _generate_indices(shape, alpha, sigma):
    dx = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    return np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

def _map_coordinates(image, indices_x, indices_y, order=1, cval=0, mode="constant"):
    result = np.copy(image)
    height, width = image.shape[0:2]
    for c in range(image.shape[2]):
        remapped_flat = scipy.ndimage.interpolation.map_coordinates(
            image[..., c],
            (indices_x, indices_y),
            order=order,
            cval=cval,
            mode=mode
        )
        remapped = remapped_flat.reshape((height, width))
        result[..., c] = remapped
    return result



def apply_salt(video: VideoData, parameters: dict) -> VideoData:
    """
    Aplica ruido de sal al video.

    Parámetros:
    - video (VideoData): El video al que se le aplicará la transformación.
    - parameters (dict): Diccionario con los parámetros de la transformación.
        - 'amount' (float): Cantidad de ruido de sal a añadir.

    Ejemplo de uso:
    parameters = {
        'amount': 0.05
    }
    """
    amount = parameters.get('amount', 0.05)

    is_PIL = isinstance(video[0], PIL.Image.Image)
    if is_PIL:
        video = [np.asarray(img) for img in video]

    data_final = []
    for img in video:
        img_shape = img.shape
        noise = np.random.randint(0, int(1.0 / amount), img_shape)
        img = np.where(noise == 0, 255, img)
        data_final.append(img.astype(np.uint8))

    if is_PIL:
        return [PIL.Image.fromarray(img) for img in data_final]
    else:
        return data_final



def apply_pepper(video: VideoData, parameters: dict) -> VideoData:
    """
    Aplica ruido de pimienta al video.

    Parámetros:
    - video (VideoData): El video al que se le aplicará la transformación.
    - parameters (dict): Diccionario con los parámetros de la transformación.
        - 'amount' (float): Cantidad de ruido de pimienta a añadir.

    Ejemplo de uso:
    parameters = {
        'amount': 0.05
    }
    """
    amount = parameters.get('amount', 0.05)

    is_PIL = isinstance(video[0], PIL.Image.Image)
    if is_PIL:
        video = [np.asarray(img) for img in video]

    data_final = []
    for img in video:
        img_shape = img.shape
        noise = np.random.randint(0, int(1.0 / amount), img_shape)
        img = np.where(noise == 0, 0, img)
        data_final.append(img.astype(np.uint8))

    if is_PIL:
        return [PIL.Image.fromarray(img) for img in data_final]
    else:
        return data_final



def apply_shear(video: VideoData, parameters: dict) -> VideoData:
    """
    Aplica un cizallamiento al video.

    Parámetros:
    - video (VideoData): El video al que se le aplicará la transformación.
    - parameters (dict): Diccionario con los parámetros de la transformación.
        - 'x' (float): Cizallamiento en el eje X.
        - 'y' (float): Cizallamiento en el eje Y.

    Ejemplo de uso:
    parameters = {
        'x': 0.2,
        'y': 0.2
    }
    """
    x = parameters.get('x', 0.2)
    y = parameters.get('y', 0.2)

    x_shear = random.uniform(-x, x)
    y_shear = random.uniform(-y, y)

    if isinstance(video[0], np.ndarray):
        rows, cols, ch = video[0].shape
        transform_mat = np.float32([[1, x_shear, 0], [y_shear, 1, 0]])
        return [cv2.warpAffine(img, transform_mat, (cols, rows)) for img in video]
    elif isinstance(video[0], PIL.Image.Image):
        return [img.transform(img.size, PIL.Image.AFFINE, (1, x_shear, 0, y_shear, 1, 0)) for img in video]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image but got list of {0}'.format(type(video[0])))