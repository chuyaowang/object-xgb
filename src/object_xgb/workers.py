import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes
from skimage import measure, morphology, segmentation
from sklearn.cluster import KMeans
from sklearn.svm import SVC


def segment_objects_worker(data: np.ndarray, orig_ndim: int, layer_type: str):
    """Core logic for object segmentation."""
    # 1. Pixel-wise segmentation
    if layer_type == 'probabilities':
        argmax_axis = 1 if orig_ndim == 3 else 0
        class_map = np.argmax(data, axis=argmax_axis)
        foreground = class_map > 0
    else:
        foreground = data > 0

    # 2. Hole filling
    if orig_ndim == 3:
        for z in range(foreground.shape[0]):
            foreground[z] = binary_fill_holes(foreground[z])
    else:
        foreground = binary_fill_holes(foreground)

    # 3. Initial Labeling
    raw_labels = measure.label(foreground)
    props = measure.regionprops(raw_labels)
    if not props:
        return raw_labels

    areas = np.array([p.area for p in props]).reshape(-1, 1)

    # 4. Automated Thresholding (K-Means + SVM)
    if len(props) > 1 and np.any(areas <= 10):
        log_areas = np.log10(areas)
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(log_areas)

        svm = SVC(kernel='linear', C=1.0)
        svm.fit(log_areas, cluster_labels)

        w = svm.coef_[0]
        b = svm.intercept_[0]
        log_threshold = -b / w
        threshold = 10**log_threshold

        if not (np.min(areas) < threshold < np.max(areas)):
            centers = 10 ** kmeans.cluster_centers_.flatten()
            threshold = np.mean(centers)

        threshold = float(np.squeeze(threshold))
        print(f'[Object XGB] Auto-detected size threshold: {threshold:.2f}')
        keep_labels = [p.label for p in props if p.area >= threshold]
    else:
        keep_labels = [p.label for p in props]

    mask = np.isin(raw_labels, keep_labels)
    filtered_labels = np.where(mask, raw_labels, 0)

    # 5. Dilation
    footprint = morphology.ball(1) if orig_ndim == 3 else morphology.disk(1)
    processed_labels = morphology.dilation(
        filtered_labels, footprint=footprint
    )

    # 6. Relabel sequentially
    final_labels, _, _ = segmentation.relabel_sequential(processed_labels)
    return final_labels


def _merge_feature_supplement(
    cached: pd.DataFrame,
    supplement: pd.DataFrame,
    supplement_cols: list[str],
) -> pd.DataFrame:
    """Fill NaN feature columns in cached with values from supplement.

    Parameters
    ----------
    cached : pd.DataFrame
        Rows from the feature cache with partial NaN feature columns.
    supplement : pd.DataFrame
        Freshly computed rows containing values for the previously-NaN columns.
    supplement_cols : list[str]
        The feature column names that supplement provides.

    Returns
    -------
    pd.DataFrame
        A copy of cached with supplement_cols filled in from supplement,
        joined on (label, slice_id).
    """
    merged = cached.copy().set_index(['label', 'slice_id'])
    sup_indexed = supplement.set_index(['label', 'slice_id'])[supplement_cols]
    for col in supplement_cols:
        if col in sup_indexed.columns:
            merged[col] = sup_indexed[col]
    return merged.reset_index()


def train_classifier_worker(
    state_objects: np.ndarray,
    intensity_data: np.ndarray,
    training_labels: np.ndarray,
    labeled_slices: list[int],
    orig_ndim: int,
    feature_extractor,
    classifier,
    full_feature_table: pd.DataFrame | None = None,
    selected_feature_groups: set[str] | None = None,
):
    """Core logic for training the classifier.

    Parameters
    ----------
    state_objects : np.ndarray
        Segmented label image (2D or 3D).
    intensity_data : np.ndarray
        Raw intensity image matching state_objects shape.
    training_labels : np.ndarray
        Manual annotation layer data.
    labeled_slices : list[int]
        Slice indices that contain manual annotations.
    orig_ndim : int
        Original image dimensionality (2 or 3).
    feature_extractor : FeatureExtractor
        Extractor instance used to compute features.
    classifier : ObjectClassifier
        Classifier pipeline to train.
    full_feature_table : pd.DataFrame or None
        Cached feature table from a previous training or prediction run.
        Rows may have partial features (NaN) if computed during prediction.
    selected_feature_groups : set[str] or None
        Feature groups computed during the last prediction phase. Used to
        determine which groups are missing from prediction-phase cached rows.
    """
    is_3d = orig_ndim == 3

    # 0. Find target labels that have annotations
    target_labels = set()
    for z in labeled_slices:
        obj_slice = state_objects[z] if is_3d else state_objects
        ann_slice = training_labels[z] if is_3d else training_labels

        mask = ann_slice > 0
        labeled_objs = np.unique(obj_slice[mask])
        for obj_id in labeled_objs:
            if obj_id > 0:
                target_labels.add(obj_id)

    if not target_labels:
        print('[Object XGB] No overlapping annotations found for the objects.')
        yield None
        return

    # 1. Categorize each labeled slice by its cache status
    #    cached_complete: all 69 feature columns present (no NaN) → reuse directly
    #    cached_partial:  only selected groups present (has NaN) → compute missing groups
    #    missing:         not in cache at all → compute all groups
    all_feature_cols = feature_extractor.get_all_feature_names()
    all_group_names = set(feature_extractor.FEATURE_GROUPS.keys())

    cached_complete: dict[int, pd.DataFrame] = {}
    cached_partial: dict[int, pd.DataFrame] = {}
    missing_slices: list[int] = []

    if full_feature_table is not None:
        for z in labeled_slices:
            mask = full_feature_table['slice_id'] == z
            if mask.any():
                rows = full_feature_table[mask].copy()
                if rows[all_feature_cols].isna().any(axis=None):
                    cached_partial[z] = rows
                else:
                    cached_complete[z] = rows
            else:
                missing_slices.append(z)
    else:
        missing_slices = list(labeled_slices)

    n_complete = len(cached_complete)
    n_partial = len(cached_partial)
    n_missing = len(missing_slices)

    if n_complete or n_partial:
        print(
            f'[Object XGB] Feature cache: {n_complete} slice(s) fully cached, '
            f'{n_partial} slice(s) partially cached, '
            f'{n_missing} slice(s) to compute.'
        )

    # 2. Determine which feature groups are missing from partial-cache slices
    missing_group_names: set[str] = set()
    missing_group_features: list[str] = []

    if cached_partial:
        if selected_feature_groups is not None:
            missing_group_names = all_group_names - selected_feature_groups
        else:
            # No record of what was computed → treat as fully missing
            missing_slices.extend(cached_partial.keys())
            cached_partial = {}

        if missing_group_names:
            missing_group_features = [
                f
                for g in missing_group_names
                for f in feature_extractor.FEATURE_GROUPS[g]
            ]

    # 3. Compute all features for fully-missing slices
    computed_complete: dict[int, pd.DataFrame] = {}
    if missing_slices:
        gen = feature_extractor.generate_features(
            state_objects,
            intensity_image=intensity_data,
            indices=missing_slices,
        )
        raw_df = None
        for val in gen:
            if isinstance(val, tuple):
                yield val
            else:
                raw_df = val

        if raw_df is not None and not raw_df.empty:
            for z in missing_slices:
                z_rows = raw_df[raw_df['slice_id'] == z]
                if not z_rows.empty:
                    computed_complete[z] = z_rows

    # 4. Compute missing feature groups for partially-cached slices
    computed_supplements: dict[int, pd.DataFrame] = {}
    partial_slice_list = list(cached_partial.keys())

    if partial_slice_list and missing_group_features:
        gen = feature_extractor.generate_features(
            state_objects,
            intensity_image=intensity_data,
            indices=partial_slice_list,
            selected_features=missing_group_features,
        )
        sup_df = None
        for val in gen:
            if isinstance(val, tuple):
                yield val
            else:
                sup_df = val

        if sup_df is not None and not sup_df.empty:
            for z in partial_slice_list:
                z_rows = sup_df[sup_df['slice_id'] == z]
                if not z_rows.empty:
                    computed_supplements[z] = z_rows

    # 5. Merge supplement into partial-cache rows to produce complete rows
    merged_partial: list[pd.DataFrame] = []
    for z, partial_rows in cached_partial.items():
        if z in computed_supplements:
            merged = _merge_feature_supplement(
                partial_rows, computed_supplements[z], missing_group_features
            )
        else:
            merged = partial_rows
        merged_partial.append(merged)

    # 6. Assemble the full feature DataFrame for all labeled slices
    all_parts = (
        list(cached_complete.values())
        + list(computed_complete.values())
        + merged_partial
    )

    if not all_parts:
        yield None
        return

    feats_df = pd.concat(all_parts, ignore_index=True)

    if feats_df.empty:
        yield None
        return

    # Emit a final progress tick when all features came from cache
    if not missing_slices and not partial_slice_list:
        n = len(labeled_slices)
        yield (n, n, f'Features loaded from cache for {n} slice(s)')

    feats_df['true_label'] = 0

    # 7. Match objects with user annotations to populate true_label
    X_train = []
    y_train = []
    feature_cols = feature_extractor.get_all_feature_names()

    for idx, row in feats_df.iterrows():
        lbl = int(row['label'])
        z = int(row['slice_id'])

        if is_3d:
            obj_mask = state_objects[z] == lbl
            ann = training_labels[z][obj_mask]
        else:
            obj_mask = state_objects == lbl
            ann = training_labels[obj_mask]

        max_cls = np.max(ann)
        if max_cls > 0:
            feats_df.at[idx, 'true_label'] = max_cls
            X_train.append(row[feature_cols].values)
            y_train.append(max_cls)

    if not X_train:
        print('[Object XGB] No overlapping annotations found for the objects.')
        yield None
        return

    # 3. Train (this handles feature selection + model training)
    classifier.train(feats_df[feature_cols], feats_df['true_label'])

    # 4. Predict for preview
    # Use classifier.predict_proba which handles the reduced features internally
    probas = classifier.predict_proba(feats_df[feature_cols])
    classes = classifier.model.classes_
    n_classes = len(classes)

    obj_shape = state_objects.shape
    if is_3d:
        prob_buffer = np.zeros(
            (obj_shape[0], n_classes, *obj_shape[1:]), dtype=np.float32
        )
    else:
        prob_buffer = np.zeros((n_classes, *obj_shape), dtype=np.float32)

    for i, (_, row) in enumerate(feats_df.iterrows()):
        lbl = int(row['label'])
        z = int(row['slice_id'])
        cls_probas = probas[i]

        if is_3d:
            mask = state_objects[z] == lbl
            for c_idx in range(n_classes):
                prob_buffer[z, c_idx][mask] = cls_probas[c_idx]
        else:
            mask = state_objects == lbl
            for c_idx in range(n_classes):
                prob_buffer[c_idx][mask] = cls_probas[c_idx]

    yield (feats_df, prob_buffer)


def apply_rf_worker(
    state_objects: np.ndarray,
    intensity_data: np.ndarray,
    orig_ndim: int,
    full_feature_table: pd.DataFrame,
    feature_extractor,
    classifier,
):
    """Core logic for applying optimized classifier to full stack."""
    classes = classifier.model.classes_
    n_classes = len(classes)
    all_collected_features = []
    is_3d = orig_ndim == 3
    obj_shape = state_objects.shape
    selected_features = classifier.selected_features

    if is_3d:
        prob_results = np.zeros(
            (obj_shape[0], n_classes, *obj_shape[1:]), dtype=np.float32
        )
    else:
        prob_results = np.zeros((n_classes, *obj_shape), dtype=np.float32)

    if is_3d:
        total_slices = obj_shape[0]
        for z in range(total_slices):
            existing_df = None
            if full_feature_table is not None:
                mask = full_feature_table['slice_id'] == z
                if mask.any():
                    existing_df = full_feature_table[mask].copy()

            if existing_df is not None:
                feats_df = existing_df
                yield (
                    z + 1,
                    total_slices,
                    f'Slice {z + 1}/{total_slices}: Reusing cached features',
                )
            else:
                # OPTIMIZED: only calculate required features
                gen = feature_extractor.generate_features(
                    state_objects,
                    intensity_image=intensity_data,
                    indices=[z],
                    selected_features=selected_features,
                )
                feats_df = None
                for val in gen:
                    if isinstance(val, tuple):
                        yield (
                            z + 1,
                            total_slices,
                            f'Slice {z + 1}/{total_slices}: {val[2]}',
                        )
                    else:
                        feats_df = val

            if feats_df is not None and not feats_df.empty:
                all_collected_features.append(feats_df)
                # classifier.predict_proba handles reduction internally
                probas = classifier.predict_proba(feats_df)

                slice_objs = state_objects[z]
                for i, (_, row) in enumerate(feats_df.iterrows()):
                    lbl = int(row['label'])
                    mask = slice_objs == lbl
                    for c_idx in range(n_classes):
                        prob_results[z, c_idx][mask] = probas[i][c_idx]

        final_table = (
            pd.concat(all_collected_features, ignore_index=True)
            if all_collected_features
            else None
        )
        yield (final_table, prob_results)
    else:
        # 2D case
        feats_df = full_feature_table
        if feats_df is None:
            # OPTIMIZED: only calculate required features
            gen = feature_extractor.generate_features(
                state_objects,
                intensity_image=intensity_data,
                selected_features=selected_features,
            )
            for val in gen:
                if isinstance(val, tuple):
                    yield (0, 1, val[2])
                else:
                    feats_df = val

        if feats_df is not None and not feats_df.empty:
            probas = classifier.predict_proba(feats_df)

            slice_objs = state_objects
            for i, (_, row) in enumerate(feats_df.iterrows()):
                lbl = int(row['label'])
                mask = slice_objs == lbl
                for c_idx in range(n_classes):
                    prob_results[c_idx][mask] = probas[i][c_idx]

        yield (feats_df, prob_results)
