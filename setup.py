"""
    Setup file for ImgDataConvertCodeGen.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 4.5.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""
from setuptools import setup

if __name__ == "__main__":

    def custom_version_scheme(version):
        if version.exact:
            return version.format_with("")
        else:
            return version.format_with("{tag}")


    try:
        setup(use_scm_version={"version_scheme": custom_version_scheme,
                               "local_scheme": "no-local-version", })
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise
