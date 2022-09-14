#!/bin/bash
cd /dist
pip install -i https://repos.infervision.com/repository/pypi/simple $(ls -1t | tail -n 1)
pip install -i https://repos.infervision.com/repository/pypi/simple pydicom
python /test_lungsegmentor.py /imagedir /result/${RESULT_FNAME}