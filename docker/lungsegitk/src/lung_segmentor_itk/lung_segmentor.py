from __future__ import absolute_import

import os
import os.path
import pickle
from collections import defaultdict
from enum import Enum

import numpy as np
import onnxruntime as rt
import pandas as pd
import SimpleITK as sitk

from .common import _LUNG_INTERSITY_THRESHOLD, logger
from .utils import getBodyMask, resample_by_ref, resample_by_spacing


class Status(Enum):
    UNKNOW = 0
    LUNG_MASK = 1
    BODY_MASK = 2
    FAILED = 3


class lungSegmentor3DConnectivitySVM(object):
    def __init__(self, model_name=None):
        if model_name is None:
            partialName = "svm181020"
            model_name = os.path.join(os.path.split(__file__)[0], partialName)
        self.model_name = model_name

        self.scaler = rt.InferenceSession("%s.scaler.onnx" % model_name)
        self.clf = rt.InferenceSession("%s.model.onnx" % model_name)

        self.scaler_in_name = self.scaler.get_inputs()[0].name
        self.scaler_out_name = self.scaler.get_outputs()[0].name

        self.clf_in_name = self.clf.get_inputs()[0].name
        self.clf_out_name = self.clf.get_outputs()[0].name

    def getLungLabelCandidate(self, connectedLabelMap, SIZE_THRESHOLD=100000, excludeEdge=True):
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(connectedLabelMap)
        imSize = np.array(connectedLabelMap.GetSize(), dtype="float")[:2]
        if excludeEdge:
            npConnectedLabelMap = sitk.GetArrayViewFromImage(connectedLabelMap)
            edgeLabels = np.concatenate(
                [
                    npConnectedLabelMap[:, 0, :].flatten(),
                    npConnectedLabelMap[:, -1, :].flatten(),
                    npConnectedLabelMap[:, :, 0].flatten(),
                    npConnectedLabelMap[:, :, -1].flatten(),
                ]
            )
            backgroundLabel = np.unique(edgeLabels)
        else:
            backgroundLabel = []
        candidateLst = []
        featureDF = defaultdict(lambda: [])
        featureDF["label"] = []
        featureDF["physicalSize"] = []
        featureDF["centeringValue"] = []
        featureDF["bndWidthRatio"] = []
        featureDF["bndPhysicalWidth"] = []
        featureDF["bndPhysicalHeight"] = []
        featureDF["isBackground"] = []
        if len(stats.GetLabels()) == 0:
            return pd.DataFrame(featureDF)
        for i in stats.GetLabels():
            if stats.GetPhysicalSize(i) < SIZE_THRESHOLD:
                continue
            if i in backgroundLabel:
                continue
            pnt = np.array(
                connectedLabelMap.TransformPhysicalPointToContinuousIndex(stats.GetCentroid(i)), dtype="float"
            )[:2]
            boundingBox = stats.GetBoundingBox(i)
            bndWidthRatio = float(boundingBox[4]) / float(boundingBox[3] + 1e-5)
            bndPhysicalWidth = float(boundingBox[3]) * connectedLabelMap.GetSpacing()[0]
            bndPhysicalHeight = float(boundingBox[4]) * connectedLabelMap.GetSpacing()[1]
            centeringValue = np.sum(((pnt - imSize / 2) / (imSize / 2)) ** 2)
            candidateLst.append([i, stats.GetPhysicalSize(i), centeringValue])
            physicalSize = stats.GetPhysicalSize(i)
            bndboxVolume = np.prod(boundingBox[3:]) * np.prod(connectedLabelMap.GetSpacing())
            featureDF["label"].append(i)
            featureDF["physicalSize"].append(physicalSize)
            featureDF["physicalSizeBndBoxRatio"].append(physicalSize / (bndboxVolume + 1e-5))
            featureDF["centeringValue"].append(centeringValue)
            featureDF["bndWidthRatio"].append(bndWidthRatio)
            featureDF["bndPhysicalWidth"].append(bndPhysicalWidth)
            featureDF["bndPhysicalHeight"].append(bndPhysicalHeight)
            featureDF["isBackground"].append(i in backgroundLabel)
        res = pd.DataFrame(featureDF)
        if len(res) == 0 or len(res["label"]) == 0:
            return res
        res.loc[:, "origCriteria"] = res["physicalSize"] / (res["centeringValue"] + 1e-5)
        origGuess = res[~res.isBackground].origCriteria.idxmax()
        res.loc[:, "origGuess"] = 0
        res.loc[origGuess, "origGuess"] = 1
        res.loc[:, "isBackground"] = res["isBackground"].astype("int")
        return res

    def getConnectedComponents(self, im, LUNG_INTERSITY_THRESHOLD):
        connectedFilter = sitk.ConnectedComponentImageFilter()
        res2 = connectedFilter.Execute(im <= LUNG_INTERSITY_THRESHOLD)
        return res2

    def generateSVMFeatures(self, candidate):
        return np.array(
            candidate[
                [
                    "bndPhysicalHeight",
                    "bndPhysicalWidth",
                    "bndWidthRatio",
                    "centeringValue",
                    "isBackground",
                    "physicalSize",
                    "physicalSizeBndBoxRatio",
                ]
            ]
        )

    def judgeEligibleLungCandidateBySVM(self, candidate):
        logger.debug("using svm model %s", self.model_name)
        features = self.generateSVMFeatures(candidate)
        scaled = self.scaler.run([self.scaler_out_name], {self.scaler_in_name: features.astype(np.float32)})[0]
        return self.clf.run([self.clf_out_name], {self.clf_in_name: scaled.astype(np.float32)})[0]

    def getLungMask(
        self,
        im,
        LUNG_INTERSITY_THRESHOLD=_LUNG_INTERSITY_THRESHOLD,
        excludeEdge=True,
        bodyMask=None,
        returnIntermediateResults=False,
    ):
        if not bodyMask is None:
            logger.info("Bodymask is provided.")
            components = self.getConnectedComponents(sitk.Mask(im, bodyMask, 1024), LUNG_INTERSITY_THRESHOLD)
        else:
            components = self.getConnectedComponents(im, LUNG_INTERSITY_THRESHOLD)

        candidate = self.getLungLabelCandidate(components, excludeEdge=excludeEdge)
        if candidate is None or len(candidate) == 0:
            logger.warning("Warning: No candidate.")
            if returnIntermediateResults:
                return None, {"components": components, "candidate": candidate}
            else:
                return None
        candidate.loc[:, "svmPredict"] = self.judgeEligibleLungCandidateBySVM(candidate)
        resLabel = np.where(candidate["svmPredict"])[0].tolist()
        logger.debug("Candidate information: \n%s" % candidate)
        if len(resLabel) == 0:
            if not np.all(candidate["origGuess"] == 0):
                logger.warning(
                    "Warning: SVM result is not the same as original criterion. We choose SVM.\n%s" % candidate
                )
            if returnIntermediateResults:
                return None, {"components": components, "candidate": candidate, "resLabel": resLabel}
            else:
                return None
        else:
            if len(resLabel) > 1:
                resLabel.sort(key=lambda x: candidate["physicalSize"][x])
                logger.warning("Warning: SVM have multiple result, we choose the largest one.\n%s" % candidate)
            if candidate["origGuess"].loc[resLabel[-1]] != 1:
                logger.warning(
                    "Warning: SVM result is not the same as original criterion. We choose SVM.\n%s" % candidate
                )
        lungLabel = candidate["label"].loc[resLabel[-1]]
        logger.debug("resLabel: %s" % resLabel)
        lungMask = components == lungLabel
        if returnIntermediateResults:
            return (
                lungMask,
                {"components": components, "candidate": candidate, "resLabel": resLabel, "lungLabel": lungLabel},
            )
        else:
            return lungMask


lungSegmentor3DSVM = lungSegmentor3DConnectivitySVM()


def getLungMask(
    im,
    LUNG_INTERSITY_THRESHOLD=_LUNG_INTERSITY_THRESHOLD,
    SPACING_UPPER_LIMIT=8,
    BODYMASK_DOWNSAMPLE_SPACING=(2, 2, 2),
    BODYMASK_SMOOTHING_DIAMETER=2,
    BODYMASK_INTENSITY_THRESHOLD=(-200, 300),
    return_intermediate_results=False,
):
    logger.info(
        "getLungMask called. Parameters: im: [%s, %s, %s], LUNG_INTERSITY_THRESHOLD: %d, SPACING_UPPER_LIMIT: %d"
        % (im.GetOrigin(), im.GetSize(), im.GetSpacing(), LUNG_INTERSITY_THRESHOLD, SPACING_UPPER_LIMIT)
    )
    spacing = np.array(im.GetSpacing())
    if np.any(spacing <= 0) or np.any(spacing >= SPACING_UPPER_LIMIT):
        logger.error("Error: pixel spacing is not valid. Lung Segmentation Failed. Spacing: %s" % str(spacing))
        if return_intermediate_results:
            return None, {}, Status.FAILED
        else:
            return None, Status.FAILED

    if return_intermediate_results:
        res, intermediateRes = lungSegmentor3DSVM.getLungMask(
            im, LUNG_INTERSITY_THRESHOLD, returnIntermediateResults=return_intermediate_results
        )
    else:
        res = lungSegmentor3DSVM.getLungMask(
            im, LUNG_INTERSITY_THRESHOLD, returnIntermediateResults=return_intermediate_results
        )
    if res is None:
        logger.warning("Lung segmentation first pass failed, trying bodyMask method.")
        logger.debug("Downsample parameters: %s" % str(BODYMASK_DOWNSAMPLE_SPACING))
        imDownsampled = resample_by_spacing(im, BODYMASK_DOWNSAMPLE_SPACING)
        logger.debug(
            "Downsampled image parameters: [%s, %s, %s]"
            % (imDownsampled.GetOrigin(), imDownsampled.GetSize(), imDownsampled.GetSpacing())
        )
        bodyMaskDownsampled = getBodyMask(imDownsampled, BODYMASK_INTENSITY_THRESHOLD, BODYMASK_SMOOTHING_DIAMETER)
        if not bodyMaskDownsampled is None:
            bodyMask = resample_by_ref(bodyMaskDownsampled, im, sitk.sitkNearestNeighbor)
            if return_intermediate_results:
                res2, intermediateRes2 = lungSegmentor3DSVM.getLungMask(
                    im,
                    LUNG_INTERSITY_THRESHOLD,
                    bodyMask=bodyMask,
                    returnIntermediateResults=return_intermediate_results,
                )
                intermediateRes["bodyMask"] = bodyMask
                intermediateRes["secondPassLungMask"] = intermediateRes2
            else:
                res2 = lungSegmentor3DSVM.getLungMask(
                    im,
                    LUNG_INTERSITY_THRESHOLD,
                    bodyMask=bodyMask,
                    returnIntermediateResults=return_intermediate_results,
                )
            if not res2 is None:
                logger.info("Lung segmentation second pass with body mask succeeded.")
                if return_intermediate_results:
                    return res2, intermediateRes, Status.LUNG_MASK
                else:
                    return res2, Status.LUNG_MASK
            else:
                logger.warning("Lung segmentation second pass failed, use body mask instead.")
                if return_intermediate_results:
                    return bodyMask, intermediateRes, Status.BODY_MASK
                else:
                    return bodyMask, Status.BODY_MASK
        else:
            logger.warning("Bodymask failed.")
            if return_intermediate_results:
                return None, intermediateRes, Status.FAILED
            else:
                return None, Status.FAILED
    else:
        logger.info("Lung segmentation succeeded.")
        if return_intermediate_results:
            return res, intermediateRes, Status.LUNG_MASK
        else:
            return res, Status.LUNG_MASK
