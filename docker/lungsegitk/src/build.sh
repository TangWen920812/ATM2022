#!/bin/bash
bash clean.sh
python setup.py bdist_wheel --universal
rm -rf build lung_segmentor_itk.egg-info
