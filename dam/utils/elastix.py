# utils/elastix.py
import numpy as np
import itk
from typing import Optional, Tuple

# TODO: maybe use more complex registration methods
def elastix_register_and_resample(
    moving: np.ndarray,   # planning [X,Y,Z]
    fixed:  np.ndarray,   # repeat   [X,Y,Z]
    *,
    spacing: Optional[Tuple[float, float, float]] = None,  # e.g., (sx,sy,sz); defaults to 1s
) -> np.ndarray:

    moving_img = itk.image_view_from_array(moving)
    fixed_img  = itk.image_view_from_array(fixed)

    # Optional voxel spacing; ITK expects (sz, sy, sx)
    if spacing is not None:
        sz, sy, sx = spacing[2], spacing[1], spacing[0]
        moving_img.SetSpacing((sz, sy, sx))
        fixed_img.SetSpacing((sz, sy, sx))

    # Build parameter object: rigid -> bspline
    param = itk.ParameterObject.New()
    param.AddParameterMap(param.GetDefaultParameterMap("rigid"))
    param.AddParameterMap(param.GetDefaultParameterMap("bspline"))

    # Functional API (simple and version-robust)
    result = itk.elastix_registration_method(
        fixed_image=fixed_img,
        moving_image=moving_img,
        parameter_object=param,
        log_to_console=False,
    )
    # Some itk-elastix versions return (image, params); normalize to image
    if isinstance(result, tuple):
        result = result[0]

    arr_zyx = itk.array_from_image(result)   # (Z,Y,X)
    return arr_zyx                   # back to [X,Y,Z]
