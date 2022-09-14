from __future__ import absolute_import

import cv2
import numpy as np
import SimpleITK as sitk

from .common import logger


def morphologyClose(im, radius=5):
    return sitk.BinaryErode(sitk.BinaryDilate(im, radius), radius)


def getVesselSegmentation(im, mask, closing=True):
    if closing:
        mask_new = morphologyClose(mask)
    else:
        mask_new = mask
    return sitk.OtsuMultipleThresholds(sitk.Mask(im, mask_new, -1024), numberOfThresholds=2) > 1


def getFissureHintByVessel(im, mask, closing=True, vesselSegmentation=None):
    if closing:
        mask_new = morphologyClose(mask)
    else:
        mask_new = mask
    if vesselSegmentation is None:
        vesselMask = getVesselSegmentation(im, mask_new, False)
    distMap = sitk.SignedMaurerDistanceMap(vesselMask)
    return sitk.Mask(sitk.IntensityWindowing(distMap, 0, 200, 0, 1), mask_new, 0)


def resample_by_spacing(im, new_spacing, interpolator=sitk.sitkLinear):
    new_spacing = np.array(new_spacing)
    scaling = np.array(new_spacing) / np.array(im.GetSpacing())
    new_size = (np.array(im.GetSize()) / scaling).astype("int")
    transform = sitk.AffineTransform(3)
    transform.SetCenter(im.GetOrigin())
    return sitk.Resample(
        im, new_size.tolist(), transform, interpolator, im.GetOrigin(), new_spacing.tolist(), im.GetDirection()
    )


def resample_by_size(im, new_size, preserve_spacing=False, interpolator=sitk.sitkLinear):
    new_size = np.array(new_size).astype("int")
    new_spacing = np.array(im.GetSize()) * np.array(im.GetSpacing()) / new_size
    transform = sitk.AffineTransform(3)
    transform.SetCenter(im.GetOrigin())
    return sitk.Resample(
        im, new_size.tolist(), transform, interpolator, im.GetOrigin(), new_spacing.tolist(), im.GetDirection()
    )


def resample_by_ref(im, refim, interpolator=sitk.sitkLinear):
    transform = sitk.AffineTransform(3)
    transform.SetCenter(im.GetOrigin())
    return sitk.Resample(im, refim, transform, interpolator)


def gaussian_lung_mask(im, downsample_spacing=2, label_interpolator=sitk.sitkNearestNeighbor):
    if type(downsample_spacing) is int or type(downsample_spacing) is float:
        downsample_spacing = [downsample_spacing, downsample_spacing, downsample_spacing]
    if downsample_spacing is None:
        downsample_spacing = [2, 2, 2]
    im_downsampled = resample_by_spacing(im, downsample_spacing)
    mask = getLungMask(sitk.DiscreteGaussian(im_downsampled))
    if mask is None:
        return None
    mask_upsampled = resample_by_ref(mask, im, label_interpolator)
    return mask_upsampled


def getLargestMask(im):
    labStat = sitk.LabelShapeStatisticsImageFilter()
    connectedComp = sitk.ConnectedComponent(im)
    labStat.Execute(connectedComp)
    vols = [(i, labStat.GetPhysicalSize(i)) for i in labStat.GetLabels()]
    vols.sort(key=lambda x: x[1])
    if len(vols) == 0:
        return None
    return connectedComp == vols[-1][0]


def getBodyMask(im, bodyThres=(-200, 300), smoothRadius=2):
    logger.debug(
        "getBodyMask called. im: [%s, %s, %s], bodyThres: %s, smoothRadius: %s"
        % (im.GetOrigin(), im.GetSize(), im.GetSpacing(), bodyThres, smoothRadius)
    )
    bodyMask = getLargestMask(sitk.And(im > bodyThres[0], im < bodyThres[1]))
    if bodyMask is None:
        logger.warning(
            "no connected component within range %d, %d can be found in this image.", bodyThres[0], bodyThres[1]
        )
        return None
    npBdMask = sitk.GetArrayFromImage(bodyMask)
    t2 = np.zeros_like(npBdMask)
    for i in range(npBdMask.shape[0]):
        t, _ = cv2.findContours(npBdMask[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        if len(t) > 0:
            cv2.drawContours(t2[i], t, -1, 1, -1)
    bodyMask = sitk.GetImageFromArray(t2)
    bodyMask.CopyInformation(im)
    if not smoothRadius is None:
        bodyMask = sitk.BinaryDilate(getLargestMask(sitk.BinaryErode(bodyMask, smoothRadius)), smoothRadius)
    return bodyMask


def getLungStartAndEndLayers(lungMask):
    if lungMask is None:
        logger.error("getLungStartAndEndLayers: None is not a valid lungMask.")
        return None
    labStat = sitk.LabelShapeStatisticsImageFilter()
    labStat.Execute(lungMask)
    if len(labStat.GetLabels()) != 1:
        logger.error("Wrong number of labels in lungMask. %s" % str(labStat.GetLabels()))
        return None
    lungLabel = labStat.GetLabels()[0]
    lungBndbox = labStat.GetBoundingBox(lungLabel)
    imDepth = lungMask.GetDepth()
    lungSt = imDepth - (lungBndbox[2] + lungBndbox[5]) + 1
    lungEd = imDepth - (lungBndbox[2]) + 1
    return (lungSt, lungEd)
