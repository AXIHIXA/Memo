import shutil

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


def main() -> None:
    shutil.rmtree('./build/', True)

    torch_extension_name: str = 'pte'

    setup(
        name=torch_extension_name,
        ext_modules=[
            CppExtension(
                name=torch_extension_name,
                sources=['./src/main.cpp'],
                include_dirs=['./include/'],
                extra_compile_args={'cxx': ['-O2'],
                                    'nvcc': ['-O2']}
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )

    shutil.copy(f'./build/lib.linux-x86_64-cpython-310/{torch_extension_name}.cpython-310-x86_64-linux-gnu.so',
                f'./build/{torch_extension_name}.cpython-310-x86_64-linux-gnu.so')


if __name__ == '__main__':
    main()
