# LIB

This description about some file at this folder. 

```
lib                         - this folder, any lib must be here
|
└───data_loader             - this module is created for process, load, compare ... anything that impact to data
|   |   __init__.py         - this file make this folder as module
│   │   process.py          - file to process data
|   └───README.md           - description about this folder and its file. This folder structure check at here too
|
└───model                   - all model AI you created must be here.
|   |   abstract            - all abstract use in model must be here
|   |   resnet              - you create a model called "resnet" then you must save it in here
|   |   __init__.py         - this file make this folder as module
│   │   config.py           - file config for model
|   |   experiment.ipynb    - file to test own model
|   └───README.md           - description about this folder and its file. This folder structure check at here too
|
└───utils                   - all logic, process or something can be duplicate will be here.
|   |   __init__.py         - this file make this folder as module
│   │   config.py           - file config for folder if need
|   |   experiment.ipynb    - file to test own file
|   |   README.md           - description about this folder and its file. This folder structure check at here too
|   └───test.py             - we have logic about test and write it here
|
|   ... (coming soon)
|   __init__.py             - this file make this folder as module
└───README.md               - description about this folder

```