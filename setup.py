from setuptools import setup

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='usv-playpen',
    version='0.2.10',
    author='@bartulem',
    author_email='mimica.bartul@gmail.com',
    classifiers=[
        'Development Status :: Alpha',
        'Intended Audience :: Experimental Neuroscientists',
        'Topic :: Running Behavioral Experiments',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='neuroscience, mouse, usv, behavior',
    package_dir={'usv-playpen': 'src'},
    package_data={'': ['*.png', '*.css', '*.mplstyle']},
    include_package_data=True,
    python_requires="==3.10.*",
    description='GUI to conduct experiments w/ multichannel audio and video acquisition',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/bartulem/usv-playpen',
    project_urls={
        'Bug Tracker': 'https://github.com/bartulem/usv-playpen/issues'
    },
    license='MIT',
    install_requires=['av==10.0.0',
                      'imgstore',
                      'librosa==0.9.2',
                      'matplotlib==3.6.0',
                      'numpy==1.23.0',
                      'numba==0.56.4',
                      'opencv-contrib-python==4.7.0.68',
                      'PIMS==0.6.1',
                      'PyQt6==6.4.2',
                      'quantumrandom==1.9.0',
                      'scipy==1.10.0',
                      'scikit-learn==1.2.1',
                      'soundfile==0.12.1',
                      'toml==0.10.2']
)
