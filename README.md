# BodyScan
Measure your body fat percentage with just a single picture!

My submission for the Global PyTorch Summer Hackathon 2019. 
Among 5%(out of 1466 participants) projects to be 
[featured in the hackathon gallery](https://devpost.com/software/bodyscan). 

## Installation
### Using virtual environment
This code has been tested on Ubuntu, PyTorch 1.2, Python 3.6 and Nvidia GTX 940MX. It is recommended to setup a python virtual environment 
and install the following packages.

1. Clone the repo
2. Install the below:
   ```
   apt-get install tk-dev python-tk
   ```

3. Activate the virutal Install the required python packages in a virtual environment

   ```
   (pytorch)$ pip3 install torch torchvision 
   (pytorch)$ pip3 install scikit-image opencv-python pandas h5py
   (pytorch)$ pip3 install cffi
   (pytorch)$ pip3 install cython
   (pytorch)$ pip3 install requests
   (pytorch)$ pip3 install future
   ```
3. Build the NMS extension

   ```
   cd lib/
   python3 setup3.py build_ext --inplace
   ```
## Usage
### Run a demo
1. `python3 measure_body.py`  
   This takes a sample picture from `data/inputs` and predicts the body fat percentage. 
### Estimate your own body fat percentage! 
1. **Instructions for taking pictures**  
The model will estimate your neck and waist circumference to predict your body fat percentage. So your neck and
waist area needs to be clearly visible in the picture. Also, the model works best when you are standing atleast 1 m 
away from the camera. Some examples:  

   **Good example**  
   
   ![Image](./data/inputs/204.jpg)
   
2. Paste your picture in `data/inputs/`
3. Run `python3 measure_body.py --image_name <name_of_your_image>.jpg`  
   Your results are shown in the screen.
   
## Working

It uses a monocular depth estimating network to produce a pixel level depth map. This was based on the CVPR 2019 paper 
'Learning the depths of moving people by watching frozen people'. At the same time, RetinaNet object detection model 
was finetuned to estimate the location of your body parts. PyTorch was used for both the networks. 
This information is combined to calculate your body measurements and body fat percentage. Some 
camera intrinsics from the exif data is also used for estimation. It uses the Navy body fat formula for calculation. 

![Process](./data/process.png)


## Acknowledgements
* Depth estimation code has been borrowed & modified from this [repo](https://github.com/google/mannequinchallenge)
  (implementation of this awesome [google AI](https://mannequin-depth.github.io/) paper). 
* Retinanet code has been borrowed & modified from [this PyTorch](https://github.com/yhenon/pytorch-retinanet) 
  implementation.
* NMS code from [here](https://github.com/huaifeng1993/NMS).
