from setuptools import setup

setup(
    name='lung_segmentor_itk',
    include_package_data=True,
    version='2.2.0',
    packages=['lung_segmentor_itk'],
    package_dir={'lung_segmentor_itk': 'lung_segmentor_itk'},
    install_requires=["numpy<1.17", "scipy<1.3", "scikit-learn<0.21", "pandas<0.25", "SimpleITK", "opencv-python-headless"],
)
