# 概述
这是一个使用传统方法提供快速肺野分割的项目。典型的分割时间在10s之内。

# Changlog
[190925] Ver 2.2.0

    1. 固定依赖包版本，正式增加python3.7.4支持。
    2. 增加python 2.7/python 3.7.4 docker测试环境和半自动测试代码。
    3. 重构代码和目录结构。
    4. 将opencv-python依赖修改为opencv-python-headless依赖，减小docker image大小。


[190704] Ver 2.1.3

    修改python3适配。


[190125] Ver 2.1.2

    初步适配python3，拒绝opencv 4.0。

[181020] Ver 2.1.1

    增加了新的SVM Feature并纳入新数据训练了新版本svm模型（ver181020）。

[180914] Ver 2.1.0

    重构大部分代码，使用logging记录日志。增加了BodyMask功能，适配插管等复杂情况。

[180327] Ver 2.0

    使用SVM训练的分类器代替原本`lungSegmentor3DConnectivity`中的逻辑。

[180327] Ver 1.2

    1. 默认getLungMask函数不再使用缓慢的`lungSegmentorRegionGrow`方法。 
    2. 优化了`lungSegmentor3DConnectivity`的速度，并且不再校验肺mask
    
# 快速使用教程

## 安装

请首先配置公司pypi源，并且删除本地的`lung_segmentor_itk`。

### Ver 2.0
```(sudo) pip install lung_segmentor_itk==2```

### Ver 1.x

```(sudo) pip install lung_segmentor_itk==1```
或者在python目录下直接通过`whl`文件安装。

## 使用

    from lung_segmentor_itk import getLungMask
    import SimpleITK as sitk
    # 用SimpleITK读入整个dicom序列
    def loadDicom(dicom_dir):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
        reader.SetFileNames(dicom_names)
        im = reader.Execute()
        return im
    im = loadDicom(dicom_dir)  # dicom_dir下应该是同一个人的一个dicom序列
    lungMask = getLungMask(im)  # lungMask是一个SimpleITK图像，其中属于肺野的为1，不属于肺野的为0
    # 如果需要转换为numpy array
    npArray = sitk.GetArrayFromImage(lungMask)
    
## 注意事项
1. 本算法设计的场景是在一个具有完整的肺的CT影像中将肺野分割出来，一般来说适用的范围是+包含完整的肺+的常规胸部扫描、胸腹连扫、或头胸连扫等情况。
2. 肺分割算法从设计来说并不适用于不完整的肺，因此并不保证在肺不完整的情况下给出正确完整的肺分割结果，也不保证这个情况下给出的分割是肺的分割（而不是肠内的气体或别的）。