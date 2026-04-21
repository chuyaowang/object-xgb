import warnings

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from skimage import feature, filters, measure


class FeatureExtractor:
    # Define feature groups for optimized calculation
    FEATURE_GROUPS = {
        'geometry': ['log_area', 'eccentricity', 'circularity'],
        'hu_moments': [f'hu_moment_{i}' for i in range(7)],
        'intensity_raw': ['raw_mean', 'raw_var', 'raw_skew', 'raw_kurt']
        + [f'raw_hist_{i}' for i in range(10)],
        'intensity_sobel': [
            'sobel_mean',
            'sobel_var',
            'sobel_skew',
            'sobel_kurt',
        ]
        + [f'sobel_hist_{i}' for i in range(10)],
        'intensity_frangi': [
            'frangi_mean',
            'frangi_var',
            'frangi_skew',
            'frangi_kurt',
        ]
        + [f'frangi_hist_{i}' for i in range(10)],
        'texture_glcm': [
            f'glcm_{p}'
            for p in [
                'contrast',
                'dissimilarity',
                'homogeneity',
                'energy',
                'correlation',
            ]
        ],
        'texture_lbp': [f'lbp_{i}' for i in range(10)],
    }

    META_COLS = ['label', 'slice_id', 'true_label']

    def __init__(self):
        pass

    @classmethod
    def get_all_feature_names(cls) -> list[str]:
        """Returns the complete list of all 69 possible feature names."""
        all_feats = []
        for group in cls.FEATURE_GROUPS.values():
            all_feats.extend(group)
        return all_feats

    def _get_required_groups(
        self, selected_features: list[str] | None
    ) -> set[str]:
        """Identifies which feature groups must be calculated."""
        if selected_features is None:
            return set(self.FEATURE_GROUPS.keys())

        required = set()
        selected_set = set(selected_features)

        for group_name, group_features in self.FEATURE_GROUPS.items():
            # If any feature in this group is selected, we must calculate the whole group
            if any(f in selected_set for f in group_features):
                required.add(group_name)

        return required

    def generate_features(
        self,
        label_image: np.ndarray,
        intensity_image: np.ndarray = None,
        indices: list[int] = None,
        target_labels: list[int] | set[int] | None = None,
        selected_features: list[str] | None = None,
    ):
        """
        Generator that yields progress (current_step, total_steps, description)
        and finally yields a pandas DataFrame of features calculated slice-by-slice.

        Parameters
        ----------
        label_image : np.ndarray
            The segmented object labels as a 2D or 3D numpy array.
        intensity_image : np.ndarray, optional
            The raw intensity image for feature calculation. Must match the shape of label_image.
            Required for intensity and texture features.
        indices : list[int], optional
            For 3D images, only generate features for these specific slice indices.
            If None, all slices in the 3D stack are processed.
        target_labels : list[int] | set[int], optional
            Specific object labels to compute features for. If provided, other objects are skipped to save time.

        Yields
        ------
        tuple | pd.DataFrame
            Intermediate yields are tuples of `(current_step, total_steps, description_string)` for progress updates.
            The final yield is a `pandas.DataFrame` containing all computed features for the specified objects.
        """
        if intensity_image is None:
            yield (0, 0, 'Intensity image required for advanced features')
            return pd.DataFrame()

        if target_labels is not None:
            target_labels = set(target_labels)

        # Resolve which groups to calculate
        required_groups = self._get_required_groups(selected_features)

        # Step 1: Normalize (0.5-99.5% clip)
        v_min, v_max = np.percentile(intensity_image, (0.5, 99.5))
        img_norm = np.clip(
            (intensity_image - v_min) / (v_max - v_min + 1e-8), 0, 1
        )

        # Unify arrays to 3D to simplify the slice-by-slice loop
        if label_image.ndim == 2:
            label_stack = label_image[np.newaxis, ...]
            int_stack = img_norm[np.newaxis, ...]
        else:
            label_stack = label_image
            int_stack = img_norm

        all_features = []
        target_indices = (
            indices if indices is not None else range(label_stack.shape[0])
        )
        total_slices = len(target_indices)

        # Compute features for one slice
        def _get_slice_features(
            l_slice: np.ndarray, i_slice: np.ndarray, z: int
        ) -> list[dict]:
            """
            Computes geometry, intensity, and texture features for all objects in a single 2D slice.

            Parameters
            ----------
            l_slice : np.ndarray
                The 2D label mask for the current slice.
            i_slice : np.ndarray
                The 2D normalized intensity image for the current slice.
            z : int
                The index of the current slice within the 3D stack (or 0 for 2D images).

            Returns
            -------
            list[dict]
                A list of dictionaries, where each dictionary contains the computed features for a single object.
            """
            # If target_labels is provided, check if any of them exist in this slice
            if target_labels is not None:
                # Fast check if any target label is in the slice
                slice_labels = np.unique(l_slice)
                if not any(lbl in target_labels for lbl in slice_labels):
                    return []

            # Precompute filters ONLY if required
            sobel_slice = (
                filters.sobel(i_slice)
                if 'intensity_sobel' in required_groups
                else None
            )
            frangi_slice = None
            if 'intensity_frangi' in required_groups:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    frangi_slice = filters.frangi(i_slice)

            props = measure.regionprops(l_slice, intensity_image=i_slice)
            slice_features = []

            for prop in props:
                lbl = prop.label
                if target_labels is not None and lbl not in target_labels:
                    continue

                feat = {'label': lbl, 'slice_id': z}

                # --- 1. Geometry & Hu Moments ---
                if (
                    'geometry' in required_groups
                    or 'hu_moments' in required_groups
                ):
                    area = prop.area
                    if 'geometry' in required_groups:
                        feat['log_area'] = np.log10(area) if area > 0 else 0
                        feat['eccentricity'] = prop.eccentricity
                        perimeter = prop.perimeter
                        feat['circularity'] = (
                            (4 * np.pi * area) / (perimeter**2)
                            if perimeter > 0
                            else 0
                        )

                    if 'hu_moments' in required_groups:
                        hu_moments = prop.moments_hu
                        for i, h in enumerate(hu_moments):
                            feat[f'hu_moment_{i}'] = (
                                np.sign(h) * np.log10(np.abs(h))
                                if h != 0
                                else 0
                            )

                # --- 2. Intensity Features ---
                min_row, min_col, max_row, max_col = prop.bbox
                mask = prop.image
                bbox = (min_row, min_col, max_row, max_col)

                def process_intensity(full_img, prefix, bbox=bbox, mask=mask):
                    r0, c0, r1, c1 = bbox
                    crop = full_img[r0:r1, c0:c1]
                    pixels = crop[mask]
                    if len(pixels) == 0:
                        d = {
                            f'{prefix}_mean': 0,
                            f'{prefix}_var': 0,
                            f'{prefix}_skew': 0,
                            f'{prefix}_kurt': 0,
                        }
                        for i in range(10):
                            d[f'{prefix}_hist_{i}'] = 0
                        return d

                    # Scale invariant histogram (np dynamic range)
                    hist, _ = np.histogram(pixels, bins=10)
                    d = {
                        f'{prefix}_mean': np.mean(pixels),
                        f'{prefix}_var': np.var(pixels),
                        f'{prefix}_skew': float(skew(pixels))
                        if len(pixels) > 2
                        else 0.0,
                        f'{prefix}_kurt': float(kurtosis(pixels))
                        if len(pixels) > 2
                        else 0.0,
                    }
                    for i, p in enumerate(hist / len(pixels)):
                        d[f'{prefix}_hist_{i}'] = float(p)
                    return d

                if 'intensity_raw' in required_groups:
                    feat.update(process_intensity(i_slice, 'raw'))
                if 'intensity_sobel' in required_groups:
                    feat.update(process_intensity(sobel_slice, 'sobel'))
                if 'intensity_frangi' in required_groups:
                    feat.update(process_intensity(frangi_slice, 'frangi'))

                # --- 3. Texture Features ---
                if (
                    'texture_glcm' in required_groups
                    or 'texture_lbp' in required_groups
                ):
                    raw_crop = i_slice[min_row:max_row, min_col:max_col]

                    if 'texture_glcm' in required_groups:
                        p_min, p_max = raw_crop.min(), raw_crop.max()
                        img_8bit = (
                            (
                                (raw_crop - p_min) / (p_max - p_min) * 255
                            ).astype(np.uint8)
                            if p_max > p_min
                            else np.zeros_like(raw_crop, dtype=np.uint8)
                        )
                        glcm = feature.graycomatrix(
                            img_8bit,
                            distances=[1],
                            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                            levels=256,
                            symmetric=True,
                            normed=True,
                        )
                        for p in [
                            'contrast',
                            'dissimilarity',
                            'homogeneity',
                            'energy',
                            'correlation',
                        ]:
                            feat[f'glcm_{p}'] = float(
                                np.mean(feature.graycoprops(glcm, p))
                            )

                    if 'texture_lbp' in required_groups:
                        lbp = feature.local_binary_pattern(
                            raw_crop, P=8, R=1, method='uniform'
                        )
                        lbp_pixels = lbp[mask]
                        hist, _ = np.histogram(
                            lbp_pixels, bins=10, range=(0, 10)
                        )
                        lbp_norm = (
                            hist / len(lbp_pixels)
                            if len(lbp_pixels) > 0
                            else np.zeros(10)
                        )
                        for i, p in enumerate(lbp_norm):
                            feat[f'lbp_{i}'] = float(p)

                slice_features.append(feat)
            return slice_features

        # Process each slice
        for i, z in enumerate(target_indices):
            l_slice = label_stack[z]
            if l_slice.max() == 0:
                continue

            yield (
                i + 1,
                total_slices,
                f'Extracting features for slice {z + 1}/{label_stack.shape[0]}...',
            )
            all_features.extend(_get_slice_features(l_slice, int_stack[z], z))

        df = pd.DataFrame(all_features)

        # Ensure consistent schema even if some features weren't calculated
        full_columns = self.META_COLS + self.get_all_feature_names()
        if not df.empty:
            df = df.reindex(columns=full_columns)
        else:
            df = pd.DataFrame(columns=full_columns)

        yield df
