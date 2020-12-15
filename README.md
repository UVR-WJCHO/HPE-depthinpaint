## HPE-depthinpaint

The test code for the paper 'Bare-hand Depth Inpainting for 3D Tracking of Hand Interacting with Object' published in ISMAR'20. 

## Requirements
'''
python==3.8.5
torch==1.7.0
numpy==1.19.4
visdom==0.1.8.9
'''

## Installation
##### Clone and install requirements
    $ git clone https://github.com/UVR-WJCHO/HPE-depthinpaint
    $ cd HPE-depthinpaint/
    $ pip3 install -r requirements.txt


## Download resources
##### Download pretrained weights



##### Download DexterHO dataset


	

## Test
Evaluates the pretrained model on DexterHO test.

    $ python test.py --weights {path_of_folder}


## Citations
If you think this code is useful for your research, consider citing:
```
@INPROCEEDINGS{
}
```

# Acknowledgements
This work was partly supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.2019-0-01270, WISE AR UI/UX Platform Development for Smartglasses) and Next-Generation Information Computing Development Program through the National Research Foundation of Korea(NRF) funded by the Ministry of Science, ICT (NRF-2017M3C4A7066316).
