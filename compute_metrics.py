import os
import numpy as np
import torch
import pandas as pd
import glob
from tqdm import tqdm
import nibabel as nib
from monai.metrics.meandice import DiceMetric
from monai.metrics import (
    compute_hausdorff_distance,
    compute_average_surface_distance,
    compute_surface_dice,
)


# compute HD95 and median distance to agreement
def compute_metrics(y_pred, y_true):
    # Flatten the predicted and true masks
    assert (
        y_pred.shape == y_true.shape
    ), f"Predicted and true masks must have the same shape. {y_pred.shape} vs {y_true.shape}"

    hd95 = compute_hausdorff_distance(
        y_pred,
        y_true,
        include_background=False,
        distance_metric="euclidean",
        percentile=95,
        directed=False,
        spacing=spacing,
    )
    msd = compute_average_surface_distance(
        y_pred,
        y_true,
        include_background=False,
        symmetric=False,
        distance_metric="euclidean",
        spacing=spacing,
    )  # mean surface distance

    assd = compute_average_surface_distance(
        y_pred,
        y_true,
        include_background=False,
        symmetric=True,
        distance_metric="euclidean",
        spacing=spacing,
    )  # average symmetric surface distance

    y_pred = y_pred[0, 1, ...].flatten()
    y_true = y_true[0, 1, ...].flatten()

    # Compute the intersection between the predicted and true masks
    intersection = np.sum(y_pred * y_true)
    union = np.sum(y_pred) + np.sum(y_true)

    # Compute the DSC
    DSC = 2 * intersection / union

    return DSC, hd95.item(), msd.item(), assd.item()


def main(df, pred_dir, gt_dir, test_set, threshold=0.5):
    pred_files = glob.glob(os.path.join(pred_dir, "*.nii.gz"))
    patient_ids = df["patient_id"].tolist()
    patient_ids = [str(pid) for pid in patient_ids]
    patient_ids = sorted(patient_ids)
    # patient_ids = [os.path.basename(f).split("_")[0] for f in pred_files]
    pred_files = [
        f for f in pred_files if os.path.basename(f).split("_")[0] in patient_ids
    ]
    assert len(pred_files) == len(
        patient_ids
    ), f"Number of patients do not match. {len(pred_files)} vs {len(patient_ids)}"
    patient_ids = [os.path.basename(f).split("_")[0] for f in pred_files]

    pred_mask_dir = os.path.join(pred_dir, f"thres_{threshold}_{test_set}")
    os.makedirs(pred_mask_dir, exist_ok=True)

    count = 0
    DSCs = []
    hd95s = []
    msds = []
    assds = []

    for patient_id in tqdm(patient_ids):
        pred_file = pred_files[count]
        gt_file = os.path.join(gt_dir, f"{patient_id}_CTV.nii.gz")
        gt = nib.load(gt_file).get_fdata().astype(np.uint8)

        pred_mask_file = os.path.join(pred_mask_dir, f"{patient_id}_CTV.nii.gz")
        if os.path.exists(pred_mask_file):
            pred = nib.load(pred_mask_file).get_fdata().astype(np.uint8)
        else:
            pred_nii = nib.load(pred_file)
            pred = pred_nii.get_fdata().astype(np.float32)
            pred = np.where(pred > threshold, 1, 0)
            mask_affine = pred_nii.affine
            mask_header = pred_nii.header
            mask_header.set_data_dtype(np.uint8)
            nib.save(nib.Nifti1Image(pred, mask_affine, mask_header), pred_mask_file)

        # concatenate the predicted and true masks
        pred = pred[np.newaxis, ...].repeat(2, axis=0)
        pred[0] = 1 - pred[1]  # background
        gt = gt[np.newaxis, ...].repeat(2, axis=0)
        gt[0] = 1 - gt[1]  # background
        DSC, hd95, msd, assd = compute_metrics(
            pred[np.newaxis, ...], gt[np.newaxis, ...]
        )
        count += 1
        DSCs.append(DSC)
        hd95s.append(hd95)
        msds.append(msd)
        assds.append(assd)

    DSCs = np.array(DSCs)
    hd95s = np.array(hd95s)
    msds = np.array(msds)
    assds = np.array(assds)
    df = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "DSC": DSCs,
            "HD95": hd95s,
            "MSD": msds,
            "ASSD": assds,
        }
    )
    df.to_excel(os.path.join(pred_mask_dir, "metrics_v3.xlsx"), index=False)


if __name__ == "__main__":
    global spacing
    spacing = [1.0, 1.0, 2.5]
    folds = [0, 1, 2, 3, 4]
    threshold = 0.5
    test_set = "external"  # external

    gt_dir = "data/voxel_median_cropped_deform/CTV"
    # voxel_median_cropped_rigid or voxel_median_cropped_deform
    # gt_dir = "data/voxel_median/CTV"

    if isinstance(folds, int):
        folds = [folds]

    pred_dirs = "Late_fusion/SwinUNETR"
    pred_files = glob.glob(os.path.join(pred_dirs, "prediction_*"))
    pred_files = sorted(pred_files)

    for pred_dir in pred_files:
        df = pd.read_excel(f"./split/RT_data_{test_set}_test_final.xlsx")

        if not os.path.exists(
            os.path.join(pred_dir, f"thres_{threshold}_{test_set}", "metrics_v3.xlsx")
        ):
            main(df, pred_dir, gt_dir, test_set, threshold)
