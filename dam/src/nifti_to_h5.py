import os
import time
import h5py
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

from tqdm.auto import tqdm
from scipy.ndimage import zoom

def _list_nii(dirpath):
    return sorted([
        os.path.join(dirpath, f)
        for f in os.listdir(dirpath)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ])

def _center_crop_or_pad(arr, target_shape):
    """Center crop; if smaller, zero-pad to target_shape."""
    z = np.zeros(target_shape, dtype=arr.dtype)
    in_shape = np.array(arr.shape)
    out_shape = np.array(target_shape)

    # compute start/end for input slice
    in_start = np.maximum((in_shape - out_shape) // 2, 0)
    in_end   = np.minimum(in_start + out_shape, in_shape)

    # compute start/end for output slice
    out_start = np.maximum((out_shape - in_shape) // 2, 0)
    out_end   = out_start + (in_end - in_start)

    z[out_start[0]:out_end[0], out_start[1]:out_end[1], out_start[2]:out_end[2]] = \
        arr[in_start[0]:in_end[0], in_start[1]:in_end[1], in_start[2]:in_end[2]]
    return z

def _resize_3d(arr, target_shape, order=1):
    """Resample volume to target_shape with ndimage.zoom."""
    factors = [t / s for t, s in zip(target_shape, arr.shape)]
    return zoom(arr, zoom=factors, order=order)

def _prep_image(vol, target_shape, method="resize"):
    if method == "crop":
        return _center_crop_or_pad(vol, target_shape).astype(np.float32, copy=False)
    else:
        # order=1 (linear) for CT images
        return _resize_3d(vol, target_shape, order=1).astype(np.float32, copy=False)

def _prep_mask(vol, target_shape, method="resize"):
    # ensure integer labels/binary stay intact
    if method == "crop":
        out = _center_crop_or_pad(vol, target_shape)
    else:
        # order=0 (nearest) for masks
        out = _resize_3d(vol, target_shape, order=0)
    # if masks are not strictly binary, keep as int16 to preserve labels
    # otherwise make them uint8
    uniq = np.unique(out)
    if uniq.size <= 4 and np.all(np.isin(uniq, [0,1])):
        return out.astype(np.uint8, copy=False)
    return out.astype(np.int16, copy=False)

def _make_anatomy_mask(vol, hu_thresh=-500, dilate_vox: int = 2):
    """
    Build an anatomy mask from CT volume:
      - Threshold at hu_thresh
      - Keep largest connected component (removes table/artifacts)
      - Fill holes
      - Dilate by N voxels to include margin around anatomy

    Parameters:
      vol: 3D numpy array (HU)
      hu_thresh: threshold for anatomy (HU > hu_thresh -> inside)
      dilate_vox: number of voxels to dilate mask (0 for no dilation)
    """
    # Hard threshold
    mask = vol > hu_thresh

    # NOTE: I'm not really sure if this is necessary. 
    # Keep largest connected component
    labels, nlabels = ndi.label(mask)
    if nlabels > 0:
        counts = np.bincount(labels.ravel())
        counts[0] = 0  # ignore background
        if counts.size > 1 and counts.max() > 0:
            largest_label = counts.argmax()
            mask = (labels == largest_label)
        else:
            mask = (labels > 0)

    # Fill holes in the component
    mask_filled = ndi.binary_fill_holes(mask)

    # Dilate if requested
    if dilate_vox > 0:
        struct = ndi.generate_binary_structure(3, 1)  # 6-connectivity
        mask_dilated = ndi.binary_dilation(
            mask_filled, 
            structure=struct,
            iterations=dilate_vox
        )
        return mask_dilated.astype(np.uint8)
    
    return mask_filled.astype(np.uint8)

def convert_nifti_to_hdf5(
    planning_dir, repeated_dir, mask_planning_dir, mask_repeated_dir, output_file,
    target_shape=(32, 32, 4), method="resize", dtype=np.float32,
    compression="gzip", compression_opts=4, max_vols=None
):
    """
    method: 'resize' (resample) or 'crop' (center crop/pad).
    target_shape: (X, Y, Z) output size for each volume.
    """
    t0 = time.time()
    planning_files       = _list_nii(planning_dir)
    repeated_files       = _list_nii(repeated_dir)
    mask_planning_files  = _list_nii(mask_planning_dir)
    mask_repeated_files  = _list_nii(mask_repeated_dir)

    n = len(planning_files)
    if not (n == len(repeated_files) == len(mask_planning_files) == len(mask_repeated_files)):
        raise ValueError(f"Counts differ: planning={len(planning_files)}, repeated={len(repeated_files)}, "
                         f"mask_planning={len(mask_planning_files)}, mask_repeated={len(mask_repeated_files)}")
    if n == 0:
        raise ValueError("No .nii/.nii.gz files found.")

    if max_vols is not None:
        n = min(n, int(max_vols))
        planning_files       = planning_files[:n]
        repeated_files       = repeated_files[:n]
        mask_planning_files  = mask_planning_files[:n]
        mask_repeated_files  = mask_repeated_files[:n]

    print(f"[INFO] Target shape: {tuple(target_shape)}, count: {n}, method: {method}")

    # Create HDF5 with tiny volumes; last axis is the case index N
    with h5py.File(output_file, "w") as h5f:
        dsets = {
            "planning":       h5f.create_dataset("planning",       shape=tuple(target_shape) + (n,),
                                                 dtype=np.float32, chunks=True, compression=compression,
                                                 compression_opts=compression_opts),
            "repeated":       h5f.create_dataset("repeated",       shape=tuple(target_shape) + (n,),
                                                 dtype=np.float32, chunks=True, compression=compression,
                                                 compression_opts=compression_opts),
            "masks_planning": h5f.create_dataset("masks_planning", shape=tuple(target_shape) + (n,),
                                                 dtype=np.int16, chunks=True, compression=compression,
                                                 compression_opts=compression_opts),
            "masks_repeated": h5f.create_dataset("masks_repeated", shape=tuple(target_shape) + (n,),
                                                 dtype=np.int16, chunks=True, compression=compression,
                                                 compression_opts=compression_opts),
            "planning_anatomy_mask":   h5f.create_dataset("planning_anatomy_mask",   shape=tuple(target_shape) + (n,),
                                                 dtype=np.uint8, chunks=True, compression=compression,
                                                 compression_opts=compression_opts),
            "repeated_anatomy_mask":   h5f.create_dataset("repeated_anatomy_mask",   shape=tuple(target_shape) + (n,),
                                                 dtype=np.uint8, chunks=True, compression=compression,
                                                 compression_opts=compression_opts),
        }

        for i in tqdm(range(n), desc="Writing HDF5 (cropped/resized)", unit="vol"):
            # Load
            p  = nib.load(planning_files[i]).get_fdata()
            r  = nib.load(repeated_files[i]).get_fdata()
            mp = nib.load(mask_planning_files[i]).get_fdata()
            mr = nib.load(mask_repeated_files[i]).get_fdata()

            # Process
            p_small  = _prep_image(p,  target_shape, method)
            r_small  = _prep_image(r,  target_shape, method)
            mp_small = _prep_mask(mp,  target_shape, method)
            mr_small = _prep_mask(mr,  target_shape, method)

            # Anatomy mask from repeated scan
            r_anatomy_mask = _make_anatomy_mask(r_small)
            p_anatomy_mask = _make_anatomy_mask(p_small)

            # Write
            dsets["planning"][..., i]       = p_small
            dsets["repeated"][..., i]       = r_small
            dsets["masks_planning"][..., i] = mp_small
            dsets["masks_repeated"][..., i] = mr_small
            dsets["planning_anatomy_mask"][..., i]   = p_anatomy_mask
            dsets["repeated_anatomy_mask"][..., i]   = r_anatomy_mask


    print(f"[DONE] {output_file}  ({time.strftime('%H:%M:%S', time.gmtime(time.time()-t0))})")

if __name__ == "__main__":
    convert_nifti_to_hdf5(
        planning_dir = "Z:/PROYECTOS_INVESTIGACION/2024-CPT-PatientModeling/002_DATASET/DAM/raw_dataset/planning",
        repeated_dir = "Z:/PROYECTOS_INVESTIGACION/2024-CPT-PatientModeling/002_DATASET/DAM/raw_dataset/repeated",
        mask_planning_dir = "Z:/PROYECTOS_INVESTIGACION/2024-CPT-PatientModeling/002_DATASET/DAM/raw_dataset/masks_planning",
        mask_repeated_dir = "Z:/PROYECTOS_INVESTIGACION/2024-CPT-PatientModeling/002_DATASET/DAM/raw_dataset/masks_repeated",
        output_file = "Z:/DOCTORADO/TGuija/RCTSynth/dam/data/mini_masked.h5",
        target_shape=(32, 32, 16),
        method="resize",              # 'resize' or 'crop'
        compression="lzf",            # faster writes; use 'gzip' for smaller files
        compression_opts=None,
        max_vols=24                   # limit number of cases for quick experiments
    )
