# DATA

This description about some file at this folder. 
Ex:
process.ipynb will call lib or func at untils or lib/datasets folder


```
data                        - this folder, any data must be here
│   README.md               - description about this folder
|
└───external                - any data we consider as being external.
│   │   data.txt            - data
|   └───README.md           - description about this external. This folder structure check at here too
|
└───internal                - data on intermediate format between raw and processed. Not raw and also not ready yet
│   │   data.txt            - data
|   └───README.md           - description about this internal. This folder structure check at here too
|
└───raw                     - here all raw data exists. This directory should be considered as read only - just |                             leave what we got as it is.
│   │   data.txt            - data
|   └───README.md           - description about this raw. This folder structure check at here too
|
└───test                    - data use to test/evaluation
│   │   data.txt            - data
|   └───README.md           - description about this test. This folder structure check at here too
|
└───train                   - data use to train
│   │   data.txt            - data
|   └───README.md           - description about this train. This folder structure check at here too
|
└───val                     - data use to validation/evaluation
    │   data.txt            - data
    └───README.md           - description about this val. This folder structure check at here too

```