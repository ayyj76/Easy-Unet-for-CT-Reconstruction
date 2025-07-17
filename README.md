# Easy-Unet-for-CT-Reconstruction

# Directory structure

├── data/

│   ├── img/

│   │   ├── patient01.npy

│   │   ├── patient02.npy

│   │   ├── patient03.npy

│   │   └── ...

│   └── label/

│       ├── patient01.npy

│       ├── patient02.npy

│       ├── patient03.npy

│       └── ...


# Data sources

The data is from the **Low Dose CT Grand Challenge - AAPM**:  
[Mayo_Grand_Challenge](https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h/folder/144226105715)

First, convert the IMA data to the numpy data type (In fact, IMA data is equivalent to DICOM data,please refer to [data conversion](https://github.com/ayyj76/Medical_data_conversion_script/tree/main/2npy))
