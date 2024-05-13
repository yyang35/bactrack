from setuptools import setup, find_packages

setup(
    name='bactrack',
    version='0.1',
    description="Cell tracking package",
    exclude_package_data={
        '': ['*.tif', '*.jpg', '*.md', '*.rst']  # Exclude all images and markdown files globally
    },
    install_requires=[
        'cellpose>=2.2.3',
        'omnipose',
        'mip',
        'numpy',
        'pandas',
        'Pillow',
        'pytest',
        'scikit-learn',
        'scikit-image',
        'torch',
        'tqdm',
        'psutil',
        'scipy',
        'pytest-cov'
    ],
    extras_require={
        "GUI": [
            'descartes',
            'ipython',
            'ipywidgets',
            'matplotlib',
            'mpl_interactions',
            'natsort',
            'networkx',
            'opencv-python-headless', 
            'PyQt6',
            'Shapely'
        ]
    }
)
