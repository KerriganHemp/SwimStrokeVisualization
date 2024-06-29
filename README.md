# Swim Tech - Swim Stroke Visualization Tool
This is the repository the holds the python machine learning code from the Mines summer 2024 Field Session. The predict.py file is thoroughly commented to help the next group understand the program flow. 

### Index

* [File Structure and Hierarchy](#file-structure-and-hierarchy)
* [Required Imports](#required-imports)
* [Functions](#functions)
* [YOLOv8 model](#yolov8-model)
* [Running the code](#running-the-code)
* [Program Output](#program-output)

## File Structure and Hierarchy
![files](https://github.com/KerriganHemp/SwimStrokeVisualization/assets/156223624/31fe0042-1ae4-4ed2-b94a-56e19b1459de)   

## Required Imports
### Ultralytics and YOLOv8 - For Pre-Trained Models
pip install: `pip install ultralytics`

import: `from ultralytics import YOLO`

Ultralytics YOLOv8 documentation: https://docs.ultralytics.com/
### OpenCV - For Computer Vision
pip install: `pip install opencv-python`

import: `import cv2`

OpenCV Python documentation: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
## Functions
All functions are in the same predict.py file.

### extract_frames()
This python method extracts frames from the input video.

Parameters:

**input_video_path** (string): The file path of the input swimming video

**frame_rate** (int): The rate at which the frames will be extracted from the input video

1 = all frames extracted

2 = every other frame extracted

3 = every third frame extracted
...
 
Returns: void

The function releases the OpenCV VideoCapture object   

Example usage: 

**Relative input video path with frame extraction rate of 1**

extract_frames('./example_video_input', 1) 

### to_video()
Compresses annotated frames back into an mp4 video format after the model makes predictions

Parameters:

**image_folder** (string): The folder location of the annotated frames that are to be compressed back to a video

**output_video_file** (string): The file name of the output video that will be saved in the current project directory
 
Returns: void
The function releases the OpenCV VideoWriter object   

Example of usage: 

extract_frames('./prediction_results', 'output_video.mp4') 

### predict() 
This is the driver method to run the YOLO model predictions. It runs the extract_frames method, makes predictions on each frame, annotates each frame with boxes, then compresses them back to an annotated video with the to_video method.

Parameters:
**input_video_path** (string): The file path of the input swimming video

Returns: void  

Example of usage: 

predict('./example_video_input') - Relative path of file saved in project directory

or 

predict("C:\\users\\john_smith\\downloads\\video_input.MOV") - Path of a video saved locally on a computer (not in project directory)

## YOLOv8 Model
Yolo saves trained models in PT (Place-Text) format (.pt file extension), to make predictions on images. The model that we trained is saved in the weights folder of the project, named 'best.pt'. This code is implemented in the predict() method:

`model = YOLO('./weights/best.pt')`

The code then runs the prediction, with an image as the source:

`results = model.predict(source=f"{image_file}, save=True)`

## Running the Code
To run the code, start by navigating to the backend directory.

Run the python predict.py file: `python predict.py`
## Program Output
I have added print statements and the time module to help you navigate the process.

The terminal will print the following as the process runs:

`Starting frame extraction`

`Saving frames to the output_dir folder`

`{number of frames} frames saved to folder in {time} seconds`

`Images are being annotated with prediction boxes`

`{number of frames} frames annotated in {time} seconds`

`Images are being written back to video format`

`Video created in {time} seconds`
