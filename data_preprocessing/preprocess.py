import SimpleITK as sitk
import numpy as np

def load_and_preprocess_ct(scan_path, target_spacing=(1.0, 1.0, 1.0)):
    image = sitk.ReadImage(scan_path)
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(osz * ospc / tspc)) 
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetInterpolator(sitk.sitkLinear)

    resampled_image = resample.Execute(image)
    return sitk.GetArrayFromImage(resampled_image)
