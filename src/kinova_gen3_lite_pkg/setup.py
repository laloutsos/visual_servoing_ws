from setuptools import find_packages, setup

package_name = 'kinova_gen3_lite_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nickl',
    maintainer_email='laloutsosnikos@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'kinova_ik_controller = kinova_gen3_lite_pkg.kinovaIKController:main',
            'kinova_close_controller = kinova_gen3_lite_pkg.kinova_close_controller:main',
            'kinova_fixed_position_controller = kinova_gen3_lite_pkg.kinova_fixed_position_controller:main'
        ],
    },
)
