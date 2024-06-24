import importlib

import pytest

packages = [
    "keras",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "tensorflow",
    "torch",
    "cv2",
]


@pytest.mark.parametrize("package_name", packages, ids=packages)
def test_import(package_name):
    """Test that certain dependencies are importable."""
    importlib.import_module(package_name)
