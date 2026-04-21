from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import napari


class ImageStateManager:
    def __init__(self):
        # Key: napari.layers.Image object, Value: dict of state
        self.image_states: dict[napari.layers.Image, dict[str, Any]] = {}
        self.current_image = None

    def init_image_state(self, layer: 'napari.layers.Layer'):
        """Initialize the state dictionary for a specific layer."""
        if layer is None or layer in self.image_states:
            return

        import napari

        path = None
        source = getattr(layer, 'source', None)
        raw_path = getattr(source, 'path', None)
        if raw_path:
            path = str(
                raw_path[0]
                if isinstance(raw_path, (list, tuple))
                else raw_path
            )

        # Determine layer type and original image dimensionality
        data = layer.data
        if isinstance(layer, napari.layers.Image):
            if np.issubdtype(data.dtype, np.floating):
                layer_type = 'probabilities'
                orig_ndim = data.ndim - 1
            else:
                layer_type = 'mask'
                orig_ndim = data.ndim
        else:
            layer_type = 'mask'
            orig_ndim = data.ndim

        self.image_states[layer] = {
            'data': data,
            'ndim': data.ndim,
            'orig_ndim': orig_ndim,
            'layer_type': layer_type,
            'name': layer.name,
            'path': path,
            'objects': None,  # Filtered + dilated label image
            'labeled_slices': [],  # Indices for 3D stacks
            'full_feature_table': None,  # Master DataFrame for all processed objects
            'selected_feature_groups': None,  # Groups computed by last prediction phase
            'training_probabilities': None,  # Buffer for training slice probs
            'prediction_probabilities': None,  # Buffer for full stack probs
        }
        print(
            f'[Object XGB] Initialized state for: {layer.name} '
            f'(Type: {layer_type}, Orig NDIM: {orig_ndim})'
        )

    def get_state(self, layer: 'napari.layers.Layer') -> dict[str, Any]:
        return self.image_states.get(layer)

    def clear_state(self, layer: 'napari.layers.Layer'):
        if layer in self.image_states:
            del self.image_states[layer]

    def reset_all(self):
        self.image_states.clear()
        self.current_image = None
