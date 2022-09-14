import os
import pickle

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def smartPickleLoader(path):
    try:
        with open(path, "r") as f:
            return pickle.load(f)
    except TypeError:
        # fix python3 loading python2 pickled str item
        with open(path, "rb") as f:
            return pickle.load(f, encoding="iso-8859-1")


def convert(sklearn_path, onnx_path, initial_type):
    mod = smartPickleLoader(sklearn_path)

    onx = convert_sklearn(mod, initial_types=initial_type)

    with open(onnx_path, "wb") as f:
        f.write(onx.SerializeToString())


if __name__ == "__main__":
    initial_type = [("svm_in", FloatTensorType([None, 7]))]

    convert(
        os.path.join(os.path.dirname(__file__), "../src/lung_segmentor_itk/svm181020.model"),
        os.path.join(os.path.dirname(__file__), "../src/lung_segmentor_itk/svm181020.model.onnx"),
        initial_type,
    )

    initial_type = [("scale_in", FloatTensorType([None, 7]))]

    convert(
        os.path.join(os.path.dirname(__file__), "../src/lung_segmentor_itk/svm181020.scaler"),
        os.path.join(os.path.dirname(__file__), "../src/lung_segmentor_itk/svm181020.scaler.onnx"),
        initial_type,
    )
