# Setup

# Write some text here

from setuptools import setup, find_packages

# Get dependencies
with open('requirements.txt', 'r', encoding='utf-8') as f:
    required = []
    for line in f:
        req = line.split('#', 1)[0].strip()
        if req and not req.startswith('--'):
            required.append(req)

    # REQUIRED = f.read().splitlines()
    REQUIRED = required

setup(name='tfx_pipeline',
      version='0.0.1',
      description='A TFX pipeline example',
      packages=find_packages(exclude=['tests']),
      install_requires=REQUIRED,
      python_requires='>=3.8',
      # get version from git tags
      setup_requires=['setuptools_scm'],
      use_scm_version=False)
