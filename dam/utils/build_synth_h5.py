#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
import numpy as np
import h5py
from scipy.ndimage import map_coordinates, gaussian_filter


def make_row_canvas_3d_xyz(X, Y, Z, band_height=2, shades=(0.35, 0.65)):
    """3D background with horizontal stripes along Y."""
    bands = (np.arange(Y) // band_height) % 2
    row_vals = (np.where(bands == 0, shades[0], shades[1])).astype(np.float32)
    vol = np.empty((X, Y, Z), dtype=np.float32)
    vol[:] = row_vals[None, :, None]
    return vol


def draw_cube_xyz(vol, x, y, z, size, value=0.0):
    """Draw a filled cube (X,Y,Z)."""
    X, Y, Z = vol.shape
    x0, y0, z0 = max(0, x), max(0, y), max(0, z)
    x1, y1, z1 = min(X, x + size), min(Y, y + size), min(Z, z + size)
    if x1 > x0 and y1 > y0 and z1 > z0:
        vol[x0:x1, y0:y1, z0:z1] = value


def make_smooth_displacement(X, Y, Z, amplitude=4.0, sigma=8.0, seed=0):
    """
    Generate a smooth random displacement field (diffeomorphic-like).
    amplitude: max displacement in voxels.
    sigma: Gaussian smoothing in voxels.
    """
    rng = np.random.default_rng(seed)
    disp = rng.normal(size=(3, X, Y, Z)).astype(np.float32)
    for i in range(3):
        disp[i] = gaussian_filter(disp[i], sigma=sigma, mode="reflect")
        disp[i] /= np.abs(disp[i]).max() + 1e-8
        disp[i] *= amplitude
    return disp  # shape (3, X, Y, Z)


def warp_volume(vol, disp):
    """
    Warp volume by displacement field disp (3, X, Y, Z).
    Uses scipy.ndimage.map_coordinates with linear interpolation.
    """
    X, Y, Z = vol.shape
    grid_x, grid_y, grid_z = np.meshgrid(
        np.arange(X), np.arange(Y), np.arange(Z), indexing="ij"
    )
    coords = np.array([
        grid_x + disp[0],
        grid_y + disp[1],
        grid_z + disp[2],
    ])
    warped = map_coordinates(vol, coords, order=1, mode="reflect")
    return warped.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Create smooth diffeomorphic synthetic 3D dataset with cube masks.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--X", type=int, default=32)
    ap.add_argument("--Y", type=int, default=32)
    ap.add_argument("--Z", type=int, default=16)
    ap.add_argument("--cube", type=int, default=8)
    ap.add_argument("--amplitude", type=float, default=4.0, help="max displacement (voxels)")
    ap.add_argument("--sigma", type=float, default=6.0, help="Gaussian smoothing of displacement")
    ap.add_argument("--n-cases", type=int, default=2)
    ap.add_argument("--gzip", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    X, Y, Z = args.X, args.Y, args.Z
    cube = args.cube
    N = args.n_cases
    comp = "gzip" if args.gzip else "lzf"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with h5py.File(args.out, "w") as h5f:
        d_planning = h5f.create_dataset(
            "planning", (X, Y, Z, N), np.float32, chunks=True, compression=comp
        )
        d_repeated = h5f.create_dataset(
            "repeated", (X, Y, Z, N), np.float32, chunks=True, compression=comp
        )
        d_mplan = h5f.create_dataset(
            "masks_planning", (X, Y, Z, N), np.int16, chunks=True, compression=comp
        )
        d_mrep = h5f.create_dataset(
            "masks_repeated", (X, Y, Z, N), np.int16, chunks=True, compression=comp
        )

        for i in range(N):
            plan = make_row_canvas_3d_xyz(X, Y, Z)
            mask_plan = np.zeros((X, Y, Z), dtype=np.float32)

            start_x = X // 4
            start_y = (Y - cube) // 2
            start_z = (Z - cube) // 2

            # draw cube both in image (value=0.0) and mask (value=1.0)
            draw_cube_xyz(plan, start_x, start_y, start_z, cube, value=0.0)
            draw_cube_xyz(mask_plan, start_x, start_y, start_z, cube, value=1.0)

            # build smooth displacement and warp both image and mask
            disp = make_smooth_displacement(X, Y, Z,
                                            amplitude=args.amplitude,
                                            sigma=args.sigma,
                                            seed=args.seed + i)
            rep = warp_volume(plan, disp)
            mask_rep = warp_volume(mask_plan, disp)
            mask_rep = (mask_rep > 0.5).astype(np.int16)

            d_planning[..., i] = plan
            d_repeated[..., i] = rep
            d_mplan[..., i] = mask_plan.astype(np.int16)
            d_mrep[..., i] = mask_rep

        h5f.attrs["note"] = "Smooth diffeomorphic synthetic dataset with cube masks"
        h5f.attrs["amplitude"] = args.amplitude
        h5f.attrs["sigma"] = args.sigma
        h5f.attrs["n_cases"] = N

    dt = time.time() - t0
    print(f"[DONE] Wrote {args.out} in {dt:.2f}s")
    print(f" planning/repeated shape: ({X},{Y},{Z},{N}) float32")
    print(" cube mask: 1 inside cube, 0 elsewhere")


if __name__ == "__main__":
    main()
