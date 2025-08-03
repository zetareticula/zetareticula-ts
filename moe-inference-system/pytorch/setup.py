from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

# Build the Node.js C++ extension
zetareticula_ext = cpp_extension.CppExtension(
    'zetareticula_ext',
    sources=['zetareticula_ext.cpp'],
    include_dirs=[
        # Node.js headers
        os.path.expanduser('~/.node-gyp/current/include/node'),
        # N-API headers
        os.path.expanduser('~/.node-gyp/current/include/node/node'),
    ],
    define_macros=[('VERSION_INFO', '1.0.0')],
    extra_compile_args=['-std=c++17']
)

setup(
    name='zetareticula',
    version='0.1.0',
    ext_modules=[zetareticula_ext],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    install_requires=[
        'torch>=1.9.0',
        'numpy>=1.20.0',
        'nodebridge>=0.2.0',
        'pybind11>=2.6.0',
    ],
    python_requires='>=3.7',
)
