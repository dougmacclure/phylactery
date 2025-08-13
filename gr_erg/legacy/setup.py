from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext = Pybind11Extension("gr_erg.legacy._phylactery_cpp", ["gr_erg/legacy/phylactery.cpp"])

setup(
    name="_phylactery_cpp",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
)
