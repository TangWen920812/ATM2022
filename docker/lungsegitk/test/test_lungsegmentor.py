#encoding: utf-8
if __name__=="__main__":
    import sys, os
    if len(sys.argv) < 2:
        print ("Infervision lung_segmentor_itk testor. \n Usage: python test_lungsegmentor.py dicom_path [output_filename]")
        sys.exit(1)

import numpy as np
import SimpleITK as sitk
import lung_segmentor_itk
import logging
import glob
import traceback
import os, os.path
import SimpleITK as sitk
import time

try:
    basestring  # attempt to evaluate basestring
    def isstr(s):
        return isinstance(s, basestring)
except NameError:
    def isstr(s):
        return isinstance(s, str)

def load_dicom(dicom_dir, return_dcm=False, return_thickness=False, coerce_error=True, logr=None):
    """用pydicom库读取dicom文件，并转换为SimpleITK Image对象。
    :param dicom_dir: str类型的路径或list类型的dcm文件列表。
    ：param return_dcm: 返回dicom列表
    ：param return_thickness: 返回各层层厚，通过SliceLocation的差值计算得出
    :param coerce_error: True表示函数内各种合规检查仅打印错误信息；False表示一旦不合规范即raise Exception
    :return: 如果return_dcm, return_thickness等参数均为False，则返回SimpleITK.Image类的图像文件
             如果任何一个return系列参数为1，则返回dict，其中可能出现的key如下
                 "sitk_img": SimpleITK.Image类，SimpleITK图片
                 "dcm_list": return_dcm为True时包含。pydicom.DataSet类的list，读入的dicom列表
                 "thickness": return_thickness为True时包含。np.array，各层的层厚，通过SliceLocation差值计算得出
                 "inequal_thickness": return_thickness为True时包含。表明是否存在层厚不等的情况。
    """
    if logr is None:
        logr = logging.getLogger()
    logr.debug("load_dicom initiated. params: %s" % str({"dicom_dir": dicom_dir, "return_dcm": return_dcm, "return_thickness": return_thickness, "coerce_error": coerce_error}))
    if isstr(dicom_dir):
        lst = glob.glob(dicom_dir + "/*")
    else:
        lst = dicom_dir
    try:
        import pydicom as dcmLib
    except:
        import dicom as dcmLib
    # 根据列表读入dicom文件
    resLst = []
    for i in lst:
        if os.path.split(i)[-1]=='download_complete' or os.path.split(i)[-1]=='_download_complete':
            continue
        try:
            curRes = dcmLib.read_file(i, force=True)
            if not "TransferSyntaxUID" in curRes.file_meta:
                curRes.file_meta.TransferSyntaxUID = dcmLib.uid.ImplicitVRLittleEndian
                logr.debug("TransferSyntaxUID missing. Coerced.")
            if "ImageType" in curRes and "ORIGINAL" in curRes.ImageType and not "LOCALIZER" in curRes.ImageType:
                curRes.AdditionalPatientHistory = os.path.split(i)[1]
                resLst.append(curRes)
        except OSError:
            logr.error("OSError while reading %s." % i)
            continue
        except IOError:
            logr.error("OSError while reading %s." % i)
            logr.error(traceback.print_exc())
            continue
    # 检查是否来自同一个Series
    series, series_count = np.unique([int(i.SeriesNumber) for i in resLst], return_counts=True)
    if series.shape[0] > 1:
        error_msg = "Error！ More than 1 series in dicom_dir. SeriesNumber: %s, Counts of SeriesNumber: %s" % (str(series), str(series_count))
        if coerce_error:
            logr.error(error_msg)
        else:
            raise Exception(error_msg)
    # 由于DICOM标准中SliceLocation并非必须项，我们用Image Position(0x20, 0x32)的最后一维计算层间距，两者一样
    # 按SliceLocation增序排序
    resLst.sort(key=lambda x: float(x[0x20, 0x32][-1]))
    # 计算空间映射参数
    pixelSpacing = [float(i) for i in resLst[0].PixelSpacing]
    origin = [float(i) for i in resLst[0].ImagePositionPatient]
    direction = [float(i) for i in resLst[0].ImageOrientationPatient]
    sliceLocationLst = np.array([float(x[0x20, 0x32][-1]) for x in resLst])
    # Z轴的方向余弦根据sliceLocation增量的符号决定
    direction += [0, 0, np.sign(sliceLocationLst[1] - sliceLocationLst[0])]
    # 检查整个序列中SliceLocation的差是否都一样
    sliceThicknessLst = np.diff(sliceLocationLst)
    sliceThicknessLst = np.append(sliceThicknessLst, sliceThicknessLst[-1])
    # bugfix[20180206]: 将sliceThicknessLst保留两位有效数字分类
    sliceThicknessLst = np.round((sliceThicknessLst * 1000)).astype("int")
    thickness, thickness_count = np.unique(sliceThicknessLst, return_counts=True)
    thickness = thickness.astype("float") / 1000
    sliceThicknessLst = sliceThicknessLst.astype("float") / 1000
    logr.info("Unique thickness: %s" % str(thickness) )
    logr.debug("Slice Thickness List: %s" % str(sliceThicknessLst))

    if thickness.shape[0] > 1:
        error_msg = "Error! Inequal slice thicknesses! Thicknesses: %s, Count of thicknesses: %s." % (str(thickness), str(thickness_count))
        inequal_thickness = True
        if not coerce_error:
            raise Exception(error_msg)
        # 存在层厚不等时，我们使用最厚的层厚作为simpleitk image的pixel spacing
        error_msg += "Using thickest thickness %.3f as z pixel spacing of SimpleITK Image." % np.max(thickness)
        pixelSpacing += [np.max(thickness)]
        logr.error(error_msg)
    else:
        inequal_thickness = False
        pixelSpacing += [thickness[0]]
    # 分配空间构造numpy数组
    npShape = [len(resLst), resLst[0].Rows, resLst[0].Columns]
    npArr = np.zeros(npShape, np.int32)
    for k, i in enumerate(resLst):
        npArr[k] = i.pixel_array * i.RescaleSlope + i.RescaleIntercept
    # 转换为SimpleITK图片并且设置空间映射参数
    logr.debug("Result sitk image metadata: %s" % str({"origin": origin, "pixelSpacing": pixelSpacing, "direction": direction, "nparrShape": npArr.shape}))
    logr.debug("nparr min - max: %d, %d" % (np.min(npArr), np.max(npArr)))
    resSITK = sitk.GetImageFromArray(npArr)
    resSITK.SetOrigin(origin)
    resSITK.SetSpacing(pixelSpacing)
    resSITK.SetDirection(direction)
    # 构建返回字典
    res_toReturn = {"sitk_img": resSITK}
    if return_dcm:
        res_toReturn["dcm_list"] = resLst
    else:
        del resLst
    if return_thickness:
        res_toReturn["thickness"] = sliceThicknessLst
        res_toReturn["inequal_thickness"] = inequal_thickness
    else:
        del sliceThicknessLst
    del sliceLocationLst, npArr
    logr.debug("load_dicom returned.")# % str(res_toReturn))
    if len(res_toReturn) == 1:
        return res_toReturn["sitk_img"]
    else:
        return res_toReturn

def generateStackedImage(imgArr, cols = 6):
    return np.concatenate([np.concatenate(imgArr[i*cols:(i+1)*cols], axis=1) for i in range(imgArr.shape[0] // cols)])

if __name__=="__main__":
    logging.basicConfig(level=logging.DEBUG)
    if len(sys.argv) >= 2:
        dicom_path = sys.argv[1]
    else:
        sys.exit(1)
    if len(sys.argv) == 3:
        output_fname = sys.argv[2]
    else:
        output_fname = os.path.split(dicom_path.rstrip("/"))[-1]
    if not os.path.isdir(dicom_path):
        print ("%s is not a path." % dicom_path)
        sys.exit(1)
    if "__version__" in lung_segmentor_itk.__dict__:
        lsi_version = lung_segmentor_itk.__version__
    else:
        lsi_version = "<2.0.0"
    print ("lung_segmentor_itk version: %s" % str(lsi_version))
    print ("Loading dicom.")
    im = load_dicom(dicom_path)
    print ("Dicom size: %s " % str(im.GetSize()))
    print ("Coordination information: %s, %s, %s" % (im.GetOrigin(), im.GetDirection(), im.GetSpacing()))
    imgArr = sitk.GetArrayFromImage(im)
    jmp=5
    cols=6
    resIm = generateStackedImage(imgArr[0:imgArr.shape[0]:jmp], cols)
    print ("Original image: %s.jpg" % output_fname)
    sitk.WriteImage(sitk.Cast(sitk.IntensityWindowing(sitk.GetImageFromArray(resIm), -1350, 150), sitk.sitkUInt8), (output_fname + ".jpg"))
    print ("Running lung_segmentor_itk.")
    stTime = time.time()
    imMask = lung_segmentor_itk.getLungMask(im)
    if imMask is None:
        print ("lung_segmentor_itk failed. Time: %.2f" % (time.time() - stTime))
        sys.exit(0)
    else:
        print ("lung_segmentor_itk succeeded. Time: %.2f" % (time.time() - stTime))
        print ("Lung mask image: %s_mask.jpg" % output_fname)
        imgMaskArr = (sitk.GetArrayFromImage(imMask) > 0).astype("uint8") * 255
        resImMask = generateStackedImage(imgMaskArr[0:imgMaskArr.shape[0]:jmp], cols)
        sitk.WriteImage(sitk.Cast(sitk.GetImageFromArray(resImMask), sitk.sitkUInt8), (output_fname + "_mask.jpg"))
        sys.exit(0)



    
