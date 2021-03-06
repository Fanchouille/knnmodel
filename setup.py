import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()


# import yaml
# with open("./environment.yml", "r") as requirements_file:
#    conda_environment = yaml.load(requirements_file, Loader=yaml.FullLoader)

# requirements = list()
# for elem in conda_environment['dependencies']:
#    if isinstance(elem, str):
#        if "python" not in elem:
#            requirements.append(elem)
#    elif isinstance(elem, dict) and 'pip' in elem:
#        requirements.extend(elem['pip'])

def parse_requirements(requirements_path):
    with open(requirements_path) as f:
        reqs = [l.strip('\n') for l in f if l.strip('\n') and not l.startswith('#')]
    if len(reqs) > 0:
        reqs = [l.split("==")[0] for l in reqs]
    return reqs


requirements = parse_requirements('requirements.txt')

setuptools.setup(
    name='knnmodel',
    version='0.0.1',
    author="Francois Valadier",
    author_email="francois.valadier@openvalue.fr",
    description="Easy to use fast kNN",
    long_description="Easy to use fast kNN",
    long_description_content_type="text/markdown",
    url="https://github.com/Fanchouille/knnmodel.git",
    package_dir={'': 'src'},
    packages=setuptools.find_packages("src"),
    install_requires=requirements,
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.7",
    ],
)
