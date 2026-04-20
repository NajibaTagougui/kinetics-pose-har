from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="kinetics-pose-har",
    version="1.0.0",
    author="Najiba Tagougui, Monji Kherallah",
    author_email="najiba.tagougui@isims.usf.tn",
    description="Privacy-preserving human action recognition using 3D pose landmarks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NajibaTagougui/kinetics-pose-har",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyarrow>=12.0.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "pose_extraction": [
            "opencv-python>=4.8.0",
            "mediapipe>=0.10.0",
        ],
        "deep_learning": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/NajibaTagougui/kinetics-pose-har/issues",
        "Source": "https://github.com/NajibaTagougui/kinetics-pose-har",
    },
)
