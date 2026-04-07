import warnings

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from skimage import feature, filters, measure


class FeatureExtractor:
    def __init__(self):
        pass

    def generate_features(
        self,
        label_image: np.ndarray,
        intensity_image: np.ndarray = None,
        indices: list[int] = None,
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

        Yields
        ------
        tuple | pd.DataFrame
            Intermediate yields are tuples of `(current_step, total_steps, description_string)` for progress updates.
            The final yield is a `pandas.DataFrame` containing all computed features for the specified objects.
        """
        if intensity_image is None:
            yield (0, 0, 'Intensity image required for advanced features')
            return pd.DataFrame()

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
            # Precompute full-slice filters to avoid bounding box edge artifacts
            sobel_slice = filters.sobel(i_slice)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                frangi_slice = filters.frangi(i_slice)

            props = measure.regionprops(l_slice, intensity_image=i_slice)
            slice_features = []

            for prop in props:
                lbl = prop.label

                # --- 1. Geometry Features ---
                area = (
                    prop.area
                )  # TODO: consider changing to area_convex or area_filled
                log_area = np.log10(area) if area > 0 else 0
                eccentricity = prop.eccentricity
                perimeter = prop.perimeter
                circularity = (
                    (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
                )

                hu_moments = prop.moments_hu
                log_hu = [
                    np.sign(h) * np.log10(np.abs(h)) if h != 0 else 0
                    for h in hu_moments
                ]

                feat = {
                    'label': lbl,
                    'slice_id': z,
                    'log_area': log_area,
                    'eccentricity': eccentricity,
                    'circularity': circularity,
                }
                for i, h in enumerate(log_hu):
                    feat[f'hu_moment_{i}'] = h

                # --- 2. Intensity Features for Original, Sobel, and Frangi Filtered Images---
                min_row, min_col, max_row, max_col = prop.bbox
                mask = prop.image

                def process_intensity(
                    full_img: np.ndarray,
                    prefix: str,
                    mask: np.ndarray,
                    bbox: tuple,
                ) -> dict:
                    """
                    Computes intensity moments and a normalized histogram for an object.

                    Parameters
                    ----------
                    full_img : np.ndarray
                        The 2D image array to extract intensity values from (e.g., raw, sobel, or frangi).
                    prefix : str
                        A prefix string to prepend to the resulting feature keys (e.g., 'raw', 'sobel').
                    mask : np.ndarray
                        A boolean mask indicating the object's exact pixel locations within the bounding box.
                    bbox : tuple
                        The bounding box (min_row, min_col, max_row, max_col) of the object.

                    Returns
                    -------
                    dict
                        A dictionary containing mean, variance, skewness, kurtosis, and a 10-bin normalized histogram.
                    """
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
                    hist_norm = hist / len(pixels)
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
                    for i, p in enumerate(hist_norm):
                        d[f'{prefix}_hist_{i}'] = float(p)
                    return d

                # Extract Color Moments and Histograms for all layers
                bbox = (min_row, min_col, max_row, max_col)
                feat.update(process_intensity(i_slice, 'raw', mask, bbox))
                feat.update(
                    process_intensity(sobel_slice, 'sobel', mask, bbox)
                )
                feat.update(
                    process_intensity(frangi_slice, 'frangi', mask, bbox)
                )

                # --- 3. Texture Features (Haralick & LBP) ---
                raw_crop = i_slice[min_row:max_row, min_col:max_col]
                p_min, p_max = raw_crop.min(), raw_crop.max()
                if p_max > p_min:
                    img_8bit = (
                        (raw_crop - p_min) / (p_max - p_min) * 255
                    ).astype(np.uint8)
                else:
                    img_8bit = np.zeros_like(raw_crop, dtype=np.uint8)

                # GLCM (Haralick)
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

                # LBP (Local Binary Pattern) - Uniform method with P=8 yields 10 bins (0 to 9); R: radius around the pixel to compute LBP
                lbp = feature.local_binary_pattern(
                    raw_crop, P=8, R=1, method='uniform'
                )
                lbp_pixels = lbp[mask]
                if len(lbp_pixels) > 0:
                    lbp_hist, _ = np.histogram(
                        lbp_pixels, bins=10, range=(0, 10)
                    )
                    lbp_hist_norm = lbp_hist / len(lbp_pixels)
                else:
                    lbp_hist_norm = np.zeros(10)

                for i, p in enumerate(lbp_hist_norm):
                    feat[f'lbp_{i}'] = float(p)

                slice_features.append(feat)
            return slice_features

        # Process each slice
        for i, z in enumerate(target_indices):
            l_slice = label_stack[z]
            i_slice = int_stack[z]

            if l_slice.max() == 0:
                continue

            yield (
                i + 1,
                total_slices,
                f'Extracting features for slice {z + 1}/{label_stack.shape[0]}...',
            )

            slice_features = _get_slice_features(l_slice, i_slice, z)
            all_features.extend(slice_features)

        yield pd.DataFrame(all_features)
