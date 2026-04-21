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


def train_classifier_worker(
    state_objects: np.ndarray,
    intensity_data: np.ndarray,
    training_labels: np.ndarray,
    labeled_slices: list[int],
    orig_ndim: int,
    feature_extractor,
    classifier,
):
    """Core logic for training the classifier."""
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

    # 1. Generate features (Calculate ALL groups for training stage)
    gen = feature_extractor.generate_features(
        state_objects, intensity_image=intensity_data, indices=labeled_slices
    )

    feats_df = None
    for val in gen:
        if isinstance(val, tuple):
            yield val
        else:
            feats_df = val

    if feats_df is None or feats_df.empty:
        yield None
        return

    feats_df['true_label'] = 0

    # 2. Match objects with user annotations
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
