from setuptools import setup

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='usv-playpen',
    version='0.7.3',
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
    packages=['usv-playpen'],
    package_dir={'usv-playpen': 'src'},
    package_data={'usv-playpen': ['*.png', '*.css', '*.mplstyle', '*.ttf']},
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
    entry_points={
        'console_scripts': [
            'usv-playpen = usv_playpen_gui:main'
        ]
    },
    install_requires=['av==10.0.0',
                      'h5py==3.11.0',
                      'imgstore',
                      'librosa==0.10.2',
                      'matplotlib==3.6.0',
                      'numpy==1.24.4',
                      'numba==0.58.1',
                      'opencv-contrib-python==4.6.0.66',
                      'PIMS==0.6.1',
                      'PyQt6==6.7.0',
                      'requests==2.32.3',
                      'scipy==1.10.0',
                      'sleap-anipose==0.1.7',
                      'scikit-learn==1.5.2',
                      'soundfile==0.12.1',
                      'toml==0.10.2']
)
