from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='fluidnet_cpp',
    ext_modules=[
        CppExtension(
            'fluidnet_cpp',
            [
                'grid.cpp',
                'advect_type.cpp',
                'calc_line_trace.cpp',
                'fluids_init.cpp'
            ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
