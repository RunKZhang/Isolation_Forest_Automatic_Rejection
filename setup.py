from setuptools import setup, find_packages

setup(
    name='IF_Reject',

    version = '0.1',

    author = 'Runkai Zhang',

    author_email='271013216@qq.com',

    description='Isolation_Forest_Automatic_Rejection',

    install_requires = [
        # 'sklearn>=1.0.1',
        'mne>=1.0.3',
        'mne_features>=0.2.1',
        'numpy>=1.23.5'
    ],

    packages=find_packages()
)