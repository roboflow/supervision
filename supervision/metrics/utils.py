from typing import Any, Dict, Tuple

import numpy as np
import numpy.typing as npt


def ensure_pandas_installed():
    try:
        import pandas  # noqa
    except ImportError:
        raise ImportError(
            "`metrics` extra is required to run the function."
            " Run `pip install 'supervision[metrics]'` or"
            " `poetry add supervision -E metrics`"
        )


def pad_mask(mask: npt.NDArray, new_shape: Tuple[int, int]) -> npt.NDArray:
    """Pad a mask to a new shape, inserting zeros on the right and bottom."""
    if len(mask.shape) != 3:
        raise ValueError(f"Invalid mask shape: {mask.shape}. Expected: (N, H, W)")

    new_mask = np.pad(
        mask,
        (
            (0, 0),
            (0, new_shape[0] - mask.shape[1]),
            (0, new_shape[1] - mask.shape[2]),
        ),
        mode="constant",
        constant_values=0,
    )

    return new_mask


def len0_like(data: npt.NDArray) -> npt.NDArray:
    """Create an empty array with the same shape as input, but with 0 rows."""
    return np.empty((0, *data.shape[1:]), dtype=data.dtype)

# def match_predictions_to_targets(
#     ious: npt.NDArray[np.float32],
#     iou_thresholds: npt.NDArray[np.float32],
# ) -> npt.NDArray[np.bool_]:
#     """
#     Find whether a prediction is a true positive at different IoU thresholds.
#     Assumes all targets and predictions belong to the same class.

#     Arguments:
#         ious (np.ndarray): The IoU values between predictions and targets. Shape (P, T),
#             where P is the number of predictions and T is the number of targets.
#         iou_thresholds (np.ndarray): The IoU thresholds to consider. Shape (N,).

#     Returns:
#         np.ndarray: A boolean array of shape (P, N), where each element is True
#             if the prediction is a true positive at the given IoU threshold.
#     """
#     correct = np.zeros((ious.shape[0], iou_thresholds.shape[0]), dtype=bool)
#     for i, iou_level in enumerate(iou_thresholds):
#         matched_indices = np.where((ious >= iou_level))

#         if matched_indices[0].shape[0]:
#             combined_indices = np.stack(matched_indices, axis=1)
#             iou_values = ious[matched_indices][:, None]
#             matches = np.hstack([combined_indices, iou_values])

#             if matched_indices[0].shape[0] > 1:
#                 matches = matches[matches[:, 2].argsort()[::-1]]

#                 _, unique_predictions_idx = np.unique(matches[:, 1], return_index=True)
#                 matches = matches[unique_predictions_idx]
#                 _, unique_targets_idx = np.unique(matches[:, 0], return_index=True)
#                 matches = matches[unique_targets_idx]

#             correct[matches[:, 1].astype(int), i] = True

#     return correct

def match_predictions_to_targets(
    predictions_classes: np.ndarray,
    target_classes: np.ndarray,
    iou: np.ndarray,
    iou_thresholds: np.ndarray
) -> np.ndarray:
    num_predictions = predictions_classes.shape[0]
    num_iou_levels = iou_thresholds.shape[0]
    correct = np.zeros((num_predictions, num_iou_levels), dtype=bool)

    correct_class = target_classes == predictions_classes[:, None]

    for i, iou_level in enumerate(iou_thresholds):
        matched_indices = np.where((iou >= iou_level) & correct_class)
        # columns: prediction_idx, target_idx

        if matched_indices[0].shape[0]:
            combined_indices = np.stack(matched_indices, axis=1)
            iou_values = iou[matched_indices][:, None]
            matches = np.hstack([combined_indices, iou_values])

            if matched_indices[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

            correct[matches[:, 0].astype(int), i] = True

    return correct

def get_tp_fn_fp_tn(
    matches: npt.NDArray[np.bool_],
    prediction_confidence: npt.NDArray[np.float32],
    prediction_class_ids: npt.NDArray[np.int_],
    true_class_ids: npt.NDArray[np.int_],
) -> npt.NDArray[np.int_]:
    """
    Get the count of true positives (TP), false positives (FP),
    false negatives (FN) and true negatives (TN) for each class.

    Arguments:
        matches (np.ndarray): A `bool` arrays of chape `(P, Thr)`, where `P`
            is the number of predictions and `Thr` is the number of IoU thresholds.
            Elements are True if the prediction is a true positive at the given IoU
            threshold. 
        prediction_confidence (np.ndarray): Prediction confidence values. Shape `(P,)`.
        prediction_class_ids (np.ndarray): Prediction class IDs. Shape `(P,)`.

    Returns:
        np.ndarray: An array of shape `(C, Thr, 5)`, where `C` is the number of classes,
            `Thr` is the number of IoU thresholds and the last dimension is the `class_id`
            and counts of TP, FP, FN, TN.
    """
    # sorted_indices = np.argsort(-prediction_confidence)
    # matches = matches[sorted_indices]
    # prediction_class_ids = prediction_class_ids[sorted_indices]

    # unique_classes = np.unique(
    #     np.concatenate([prediction_class_ids, true_class_ids]))
    # num_classes = unique_classes.shape[0]
    # num_iou_thresholds = matches.shape[1]

    # result_array = np.zeros((num_classes, num_iou_thresholds, 5), dtype=int)

    # for class_idx, class_id in enumerate(unique_classes):
    #     is_class_pred = prediction_class_ids == class_id
    #     is_class_true = true_class_ids == class_id
        
    #     for thr_idx in range(num_iou_thresholds):
    #         class_matches = matches[is_class_pred, thr_idx]
            
    #         true_positives = class_matches.sum()
    #         false_positives = is_class_pred.sum() - true_positives
    #         false_negatives = is_class_true.sum() - true_positives
    #         true_negatives = len(true_class_ids) - is_class_true.sum() - false_positives

    #         result_array[class_idx, thr_idx] = np.array([
    #             class_id, true_positives, false_positives, false_negatives, true_negatives
    #         ])

    # TODO: sort outside
    # return result_array
    sorted_indices = np.argsort(-prediction_confidence)
    matches = matches[sorted_indices]
    prediction_class_ids = prediction_class_ids[sorted_indices]

    unique_classes, class_counts = np.unique(true_class_ids, return_counts=True)
    num_classes = unique_classes.shape[0]
    num_iou_thresholds = matches.shape[1]

    result_array = np.zeros((num_classes, num_iou_thresholds, 5), dtype=int)

    for class_idx, class_id in enumerate(unique_classes):
        is_class = prediction_class_ids == class_id
        total_true = class_counts[class_idx]
        total_prediction = is_class.sum()

        if total_prediction == 0 or total_true == 0:
            continue

        true_positives = matches[is_class].cumsum(0)
        false_positives = (1 - matches[is_class]).cumsum(0)
        false_negatives = total_true - true_positives
        true_negatives = len(true_class_ids) - total_true - false_positives

        # print(true_positives.shape)
        # print(class_idx, true_positives, false_positives, false_negatives, true_negatives)
        result_array[class_idx] = np.stack([
            class_idx,
            true_positives,
            false_positives,
            false_negatives,
            true_negatives
        ], axis=1)

    return result_array



# def get_precision(
#     id_tp_fp_fn_tn: npt.NDArray[np.int_],
# ) -> npt.NDArray[np.float32]:
#     """
#     Given an array of `class_id, TP, FP, FN, TN` counts, for multiple IoU thresholds,
#     compute the precision for each class and IoU threshold.

#     Arguments:
#         id_tp_fp_fn_tn (np.ndarray): An array of shape `(C, Thr, 5)`, where `C` is the number
#             of classes, `Thr` is the number of IoU thresholds and the last dimension is the `class_id`
#             and counts of TP, FP, FN, TN.

#     Returns:
#         np.ndarray: An array of shape `(C, Thr)`, where `C` is the number of classes, `Thr`
#             is the number of IoU thresholds, and the elements are the precision values.
#     """
#     eps = 1e-16
#     num_classes, num_iou_thresholds, _ = id_tp_fp_fn_tn.shape
#     precision = np.zeros((num_classes, num_iou_thresholds), dtype=np.float32)

#     tp = id_tp_fp_fn_tn[:, :, 1]
#     fp = id_tp_fp_fn_tn[:, :, 2]
#     precision = tp / (tp + fp + eps)

#     return precision

# def get_recall(
#     id_tp_fp_fn_tn: npt.NDArray[np.int_],
# ) -> npt.NDArray[np.float32]:
#     """
#     Given an array of `class_id, TP, FP, FN, TN` counts, for multiple IoU thresholds,
#     compute the recall for each class and IoU threshold.

#     Arguments:
#         id_tp_fp_fn_tn (np.ndarray): An array of shape `(C, Thr, 5)`, where `C` is the number
#             of classes, `Thr` is the number of IoU thresholds and the last dimension is the `class_id`
#             and counts of TP, FP, FN, TN.

#     Returns:
#         np.ndarray: An array of shape `(C, Thr)`, where `C` is the number of classes, `Thr`
#             is the number of IoU thresholds, and the elements are the recall values.
#     """
#     eps = 1e-16
#     num_classes, num_iou_thresholds, _ = id_tp_fp_fn_tn.shape
#     recall = np.zeros((num_classes, num_iou_thresholds), dtype=np.float32)

#     tp = id_tp_fp_fn_tn[:, :, 1]
#     fn = id_tp_fp_fn_tn[:, :, 3]
#     recall = tp / (tp + fn + eps)

#     return recall



# def compute_mAP(
#     predictions_confidence: npt.NDArray[np.float32],
#     predictions_class_ids: npt.NDArray[np.int_],
#     targets_class_ids: npt.NDArray[np.int_],
#     ious: npt.NDArray[np.float32],
#     iou_thresholds: npt.NDArray[np.float32],
# ) -> None:
#     """
#     Compute the mean Average Precision (mAP)

#     Arguments:
#         predictions_confidence (np.ndarray): Prediction confidence values. Shape `(P,)`.
#         predictions_class_ids (np.ndarray): Prediction class IDs. Shape `(P,)`.
#         targets_class_ids (np.ndarray): Target class IDs. Shape `(T,)`.
#         ious (np.ndarray): The IoU values between predictions and targets. Shape `(P, T)`.
#         iou_thresholds (np.ndarray): The IoU thresholds to consider. Shape `(Thr,)`.
#     """
#     matches = match_predictions_to_targets(ious, iou_thresholds)
#     # (C, Thr, 5)

#     id_tp_fp_fn_tn = get_tp_fn_fp_tn(
#         matches, predictions_confidence, predictions_class_ids, targets_class_ids
#     )
#     # (C, Thr, 5)

    
#     precision = get_precision(
#         id_tp_fp_fn_tn[:, :, 1],
#         id_tp_fp_fn_tn[:, :, 2]
#     )
#     recall = get_recall(
#         id_tp_fp_fn_tn[:, :, 1],
#         id_tp_fp_fn_tn[:, :, 3]
#     )

#     ap = get_average_precision(precision, recall)

#     print(ap)

#     # for class_idx in classes:
#     #     for thr_idx in range(num_iou_thresholds):
            

#     #         if tp + fp == 0:
#     #             continue

#     #         precision = get_precision(tp, fp)
#     #         recall = get_recall(tp, fn)

#     #         average_precisions[class_idx, thr_idx] = get_average_precision(precision, recall)

#     # mAP = np.mean(average_precisions)
#     # return {
#     #     "mAP": mAP,
#     #     "AP": average_precisions,
#     # }




def match(
    prediction_classes: np.ndarray,
    target_classes: np.ndarray,
    ious: np.ndarray,
    iou_threshold: float
) -> np.ndarray:
    """
    Find the prediction and target index pairs that match at different IoU thresholds.

    Arguments:
        prediction_classes: np.ndarray: The class IDs of the predictions. Shape (P,).
        target_classes: np.ndarray: The class IDs of the targets. Shape (T,).
        ious (np.ndarray): The IoU values between predictions and targets. Shape (P, T),
            where P is the number of predictions and T is the number of targets.
        iou_threshold (float): The IoU threshold, filtering out matches.

    Returns:
        np.ndarray: An array of shape (M, 2), where each element is a pair of
            prediction-target indices.
    """

    correct_class = prediction_classes[:, None] == target_classes

    matched_idx = np.where((ious >= iou_threshold) & correct_class)
    matched_row_idx, matched_col_idx = matched_idx
    matches = np.stack((matched_row_idx, matched_col_idx), axis=1)

    if len(matched_row_idx) > 1:
        iou_values = ious[matched_idx]
        matches = matches[iou_values.argsort()[::-1]]
        _, unique_pred_idx = np.unique(matches[:, 0], return_index=True)
        matches = matches[unique_pred_idx]
        _, unique_target_idx = np.unique(matches[:, 1], return_index=True)
        matches = matches[unique_target_idx]

    return matches

def cumulative_confusion_matrix(
    class_id: int,
    pred_class_ids :np.ndarray,
    true_class_ids: np.ndarray,
    matches: np.ndarray
):
    """
    For a given class, compute the TP, FP, FN values (but not TN), taking into
    account the matches between predictions and targets.

    Arguments:
        class_id (int): The class ID to compute the values for.
        pred_class_ids (np.ndarray): The class IDs of the predictions. Shape (P,).
        true_class_ids (np.ndarray): The class IDs of the targets. Shape (T,).
        matches (np.ndarray): The matched prediction-target indices. Shape (M, 2).

    Returns:
        np.ndarray: An array of shape (M, 4, ), for each match containing the class_id,
        and cumulative TP, FP, FN values. These values are cumulative, computed by
        computing confusion matrix value with matching predictions added one-at-a-time.
    """
    # return result_array
    is_class = pred_class_ids[matches[:, 0]] == class_id
    matches = matches[is_class]

    result_list = []
    iter_range = [0] + list(range(1, len(matches)))  # run even if len(matches) == 0
    for i in reversed(iter_range):
        current_matches = matches if i == 0 else matches[:-i]
        true_positives = len(current_matches)
        false_positives = np.sum(pred_class_ids == class_id) - i - true_positives
        false_negatives = np.sum(true_class_ids == class_id) - true_positives

        result_array = np.array([class_id, true_positives, false_positives, false_negatives], dtype=int)
        result_list.append(result_array)

    stacked = np.stack(result_list)
    return stacked


def get_precision(tp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    return tp / (tp + fp + 1e-16)

def get_recall(tp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    return tp / (tp + fn + 1e-16)

def get_average_precision(
    precision_values: np.ndarray,
    recall_values: np.ndarray,
) -> float:
    """
    Compute the average precision (AP) from precision and recall values at different
    IoU thresholds, using the 101-point interpolation method.

    Arguments:
        precision_values (np.ndarray): Precision values for each class and IoU threshold.
            Shape `(Thr, )`, where `Thr` is the number of IoU thresholds.
        recall_values (np.ndarray): Recall values for each class and IoU threshold.
            Shape `(Thr, )`, where `Thr` is the number of IoU thresholds.

    Returns:
        float: The average precision value.
    """
    extended_recall = np.concatenate(([0.0], recall_values, [1.0]))
    extended_precision = np.concatenate(([1.0], precision_values, [0.0]))
    max_accumulated_precision = np.flip(
        np.maximum.accumulate(np.flip(extended_precision))
    )
    interpolated_recall_levels = np.linspace(0, 1, 101)
    interpolated_precision = np.interp(
        interpolated_recall_levels, extended_recall, max_accumulated_precision
    )
    average_precision = np.trapz(interpolated_precision, interpolated_recall_levels)
    return float(average_precision)

def compute_mAP(
    pred_class_ids: np.ndarray,
    pred_confidence: np.ndarray,
    true_class_ids: np.ndarray,
    ious: np.ndarray
) -> dict:
    """
    Compute the mean Average Precision (mAP) from predictions and targets.

    Arguments:
        pred_class_ids (np.ndarray): The class IDs of the predictions. Shape (P,).
        pred_confidence (np.ndarray): The confidence values of the predictions. Shape (P,).
        true_class_ids (np.ndarray): The class IDs of the targets. Shape (T,).
        ious (np.ndarray): The IoU values between predictions and targets. Shape (P, T).

    Returns:
        float: The mean Average Precision value.
    """
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    unique_class_ids = set(pred_class_ids) | set(true_class_ids)

    # TODO: test if global sorting is enough
    sorted_indices = np.argsort(-pred_confidence)
    pred_class_ids = pred_class_ids[sorted_indices]
    pred_confidence = pred_confidence[sorted_indices]
    ious = ious[sorted_indices]

    ap_per_class = np.zeros((len(unique_class_ids), len(iou_thresholds)))

    matches_by_threshold = []
    for threshold in iou_thresholds:
        matches = match(pred_class_ids, true_class_ids, ious, threshold)
        matches_by_threshold.append(matches)

    for class_idx, class_id in enumerate(unique_class_ids):
        
        
        
        if class_id != 7:
            continue
        
        
        
        tp_fp_fn = []
        for thr_idx, threshold in enumerate(iou_thresholds):
            matches = matches_by_threshold[thr_idx]
            conf_mtx = cumulative_confusion_matrix(class_id, pred_class_ids, true_class_ids, matches)
            tp_fp_fn.append(conf_mtx)
        
        for val in tp_fp_fn:
            print(val.shape)
        tp_fp_fn = np.vstack(tp_fp_fn)

        if class_id == 7:
            print(f"NEW conf mtx:")
            print(tp_fp_fn)
            # tp_fp_fn[thr_idx] = conf_mtx

        # if class_id == 7:
        #     print(f"NEW conf mtx:")
        #     print(tp_fp_fn)

        precision = get_precision(tp_fp_fn[:, 1], tp_fp_fn[:, 2])
        recall = get_recall(tp_fp_fn[:, 1], tp_fp_fn[:, 3])
        ap = get_average_precision(precision, recall)
        ap_per_class[class_idx, thr_idx] = ap
        # precision = get_precision(conf_mtx[:, 1], conf_mtx[:, 2])
        # recall = get_recall(conf_mtx[:, 1], conf_mtx[:, 3])
        # ap = get_average_precision(precision, recall)
        # ap_per_class[class_idx, thr_idx] = ap

    mAP_50 = np.mean(ap_per_class[:, 0])
    mAP_75 = np.mean(ap_per_class[:, 5])
    mAP_50_95 = np.mean(ap_per_class)

    return {
        "mAP_50": float(mAP_50),
        "mAP_75": float(mAP_75),
        "mAP_50_95": float(mAP_50_95),
        "AP": ap_per_class
    }