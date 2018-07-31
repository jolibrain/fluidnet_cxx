from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='advection_cpp',
    ext_modules=[
        CppExtension(
            'advection_cpp',
            [
                'grid.cpp',
                'advect_type.cpp',
                'calc_line_trace.cpp',
                'advection.cpp',
            ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
