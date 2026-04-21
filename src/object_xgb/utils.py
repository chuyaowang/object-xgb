from pathlib import Path

import numpy as np


def get_class_map(prob_map: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Derive integer class map from probability map using argmax.
    Ensures background pixels (where all probabilities are 0) remain 0.

    Parameters
    ----------
    prob_map : np.ndarray
        The probability map as a (C, Y, X) or (Z, C, Y, X) numpy array.
    classes : np.ndarray
        The unique class labels from the trained classifier.

    Returns
    -------
    np.ndarray
        The integer class map as a (Y, X) or (Z, Y, X) numpy array.
    """
    if prob_map is None:
        return None

    # prob_map: (Z, C, Y, X) for 3D, (C, Y, X) for 2D
    is_3d = prob_map.ndim == 4
    argmax_axis = 1 if is_3d else 0

    max_probs = np.max(prob_map, axis=argmax_axis)
    class_indices = np.argmax(prob_map, axis=argmax_axis)

    # Map indices to actual class values
    class_map = classes[class_indices].astype(np.uint8)

    # Zero out background (where all probabilities were 0)
    class_map[max_probs == 0] = 0

    return class_map


def get_save_directory(layer, layer_name: str) -> Path:
    """Determine the save directory based on the layer's source path or name.

    Parameters
    ----------
    layer : napari.layers.Layer
        The napari layer to derive the path from.
    layer_name : str
        The name of the layer to use if no path is found.

    Returns
    -------
    Path
        The absolute path to the export directory.
    """
    path = None
    source = getattr(layer, 'source', None)
    raw_path = getattr(source, 'path', None)
    if raw_path:
        path = str(
            raw_path[0] if isinstance(raw_path, (list, tuple)) else raw_path
        )

    if path:
        save_dir = Path(path).parent / layer_name
    else:
        save_dir = Path.home() / layer_name

    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir
