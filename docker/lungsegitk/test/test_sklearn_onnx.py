import os
import sys

import numpy as np
import onnxruntime as rt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))
try:
    from tools.onnx_converter import smartPickleLoader
except Exception:
    raise


def test_clf():
    sklearn_clf = smartPickleLoader(
        os.path.join(os.path.dirname(__file__), "../src/lung_segmentor_itk/svm181020.model")
    )
    onnx_clf = rt.InferenceSession(
        os.path.join(os.path.dirname(__file__), "../src/lung_segmentor_itk/svm181020.model.onnx")
    )

    clf_in_name = onnx_clf.get_inputs()[0].name
    clf_out_name = onnx_clf.get_outputs()[0].name

    vec = np.random.rand(10000, 7) * 1 - 1.7
    assert (sklearn_clf.predict(vec) == onnx_clf.run([clf_out_name], {clf_in_name: vec.astype(np.float32)})[0]).all()


def test_scaler():
    sklearn_scaler = smartPickleLoader(
        os.path.join(os.path.dirname(__file__), "../src/lung_segmentor_itk/svm181020.scaler")
    )
    onnx_scaler = rt.InferenceSession(
        os.path.join(os.path.dirname(__file__), "../src/lung_segmentor_itk/svm181020.scaler.onnx")
    )

    scaler_in_name = onnx_scaler.get_inputs()[0].name
    scaler_out_name = onnx_scaler.get_outputs()[0].name

    vec = np.random.rand(10000, 7) * 100 - 200
    assert (
        np.max(
            np.abs(
                sklearn_scaler.transform(vec)
                - onnx_scaler.run([scaler_out_name], {scaler_in_name: vec.astype(np.float32)})[0]
            )
        )
        < 1e-3
    )
