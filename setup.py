from setuptools import setup

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='usv-playpen',
    version='0.8.2',
    author='@bartulem',
    author_email='mimica.bartul@gmail.com',
    classifiers=[
        'Development Status :: Alpha',
        'Intended Audience :: Experimental Neuroscientists',
        'Topic :: Running ana Analyzing Neural and Behavioral Experiments',
        'License :: OSI Approved :: GNU General Public License v3.0',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='neuroscience, mouse, usv, behavior, social, courtship',
    packages=['usv_playpen', 'usv_playpen.analyses', 'usv_playpen.visualizations', 'usv_playpen._tests', 'usv_playpen.other'],
    package_dir={'usv_playpen': 'src'},
    package_data={'usv_playpen': ['img/*.png', 'fonts/*.ttf', '_config/*', '_parameter_settings/*.json', 'other/cluster/*/*', 'other/playback/*.py', 'other/synchronization/*.ino']},
    include_package_data=True,
    python_requires="==3.10.*",
    description='GUI to conduct, process and analyze experiments w/ multichannel e-phys, audio and video acquisition',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/bartulem/usv-playpen',
    project_urls={
        'Bug Tracker': 'https://github.com/bartulem/usv-playpen/issues'
    },
    license='MIT',
    entry_points={
        'console_scripts': [
            'usv-playpen = usv_playpen.usv_playpen_gui:main'
        ]
    },
    install_requires=['astropy==6.1.7',
                      'av==10.0.0',
                      'h5py==3.11.0',
                      'imgstore',
                      'joblib==1.4.2',
                      'librosa==0.10.2',
                      'matplotlib==3.10.0',
                      'noisereduce==3.0.3',
                      'numpy==1.24.4',
                      'numba==0.58.1',
                      'opencv-contrib-python==4.6.0.66',
                      'pandas==2.2.3',
                      'PIMS==0.6.1',
                      'PyQt6==6.7.0',
                      'polars==1.28.0',
                      'pydub==0.25.1',
                      'requests==2.32.3',
                      'scipy==1.10.0',
                      'sleap-anipose==0.1.7',
                      'scikit-learn==1.5.2',
                      'toml==0.10.2',
                      'tqdm==4.67.1']
)
