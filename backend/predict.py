from ultralytics import YOLO
import glob
import cv2
import os
import shutil
import time

"""
Extracts frames from the input video

Parameters:
input_video_path (string): The file path of the input swimming video
frame_rate (int): The rate at which the frames will be extracted from the input video
    1 = all frames extracted
    2 = every other frame extracted
    3 = every third frame extracted
    ...
 
Returns: 
void: Releases the OpenCV VideoCapture object   

Example of usage: 
Relative input video path with frame extraction rate of 1
extract_frames('./example_video_input', 1) 
"""
def extract_frames(input_video_path, frame_rate):  
    print("Starting frame extraction")
    # Create a VideoCapture object with OpenCV
    cap = cv2.VideoCapture(input_video_path)

    # Directory for extracted frames 
    output_dir = './output_frames'
    
    # Delete the ouput_dir folder if it exists
    # This is to avoid processing frames from previous runs
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print("Creating output_dir folder")
    
    # Create the output directory for the extracted frames 
    os.makedirs(output_dir)  
    
    frame_count = 0
    saved_frame_count = 0
    
    #Capture each frame of the input video with OpenCV
    print("Saving frames to the output_dir folder")
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Change the mod value depending on how many frames are to be extracted
        # mod 1 extracts all frames, mod 2 extracts every other frame, etc.
        if frame_count % frame_rate == 0: 
            frame_path = os.path.join(output_dir, f'frame_{saved_frame_count:04d}.jpg')
            cv2.imwrite(frame_path, frame)
            saved_frame_count += 1
        frame_count += 1
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"{saved_frame_count} frames saved to folder in {duration} seconds")
    
    # Release the video capture object
    cap.release()

"""
Compresses annotated frames back into an mp4 video format after the 
model makes predictions

Parameters:
image_folder (string): The folder location of the annotated frames that are to be compressed back to a video
output_video_file (string): The file name of the output video that will be saved in the current project directory
 
Returns: 
void: Releases the OpenCV VideoWriter object   

Example of usage: 
extract_frames('./prediction_results', 'output_video.mp4') 
"""
def to_video(image_folder, output_video_file):
    # Recursively get all images from the folder and subfolders
    images = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            # If a file is of type .jpg or .png, add it to the images list
            if file.endswith(".jpg") or file.endswith(".png"):
                images.append(os.path.join(root, file))

    # Check if any images are found
    if not images:
        print("No images found in the specified directory.")
        return

    # Read the first image to get dimensions
    first_image_path = images[0]
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Could not read the first image at {first_image_path}")
        return
    height, width, layers = frame.shape

    # Define the codec and create a VideoWriter object
    # Output video is assigned to a mp4 video type as of now
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_file, fourcc, 30, (width, height))

    # Loop through images and write them to the video
    print("Images are being written back to video format")
    start_time = time.time()
    for image_path in images:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Could not read image at {image_path}")
            continue
        video.write(frame)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Video created in {duration} seconds")
    
    # Release the video writer object
    video.release()

"""
This is the driver method to run the YOLO model predictions.
Runs the extract_frames method, makes predictions on each frame, annotates each frame
with boxes, then compresses them back to an annotated video with the to_video method.

Parameters:
input_video_path (string): The file path of the input swimming video

Returns: 
void  

Example of usage: 
predict('./example_video_input') - Relative path of file saved in project directory
or 
predict("C:\\users\\john_smith\\downloads\\video_input.MOV") - Path of a video saved locally on a computer (not in project directory)
"""
def predict(input_video_path):
    # Call the extract_frames function
    # The 1 parameter value means the method will extract all frames of the video
    extract_frames(input_video_path, 1)

    # This is the trained model that we worked on - it is saved locally in the weights folder
    model = YOLO('./weights/best.pt')

    # Output directory for individual extracted images from input video
    image_dir = './output_frames'

    # Creates a list containing the paths of all the .jpg files in the image_dir directory (output_frames folder)
    image_files = glob.glob(f"{image_dir}/*.jpg")

    # Output directory for extracted frames with annotated prediction boxes (output from the YOLO model)
    # These files have the exact same naming convention as the frame images in the image_dir folder
    result_dir = './prediction_results'
    
    # Delete the result_dir folder if it exists
    # This is to avoid processing frames from previous runs
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    
    # Create the output directory for the annotated frames
    os.makedirs(result_dir)  

    # Loop through all images in the image_files list (images in the output_frames folder)
    print("Images are being annotated with prediction boxes")
    start_time = time.time()
    
    for image_file in image_files:
        # Single instance of a frame from the video
        image = cv2.imread(image_file)
        
        # Checks to make sure the image file is read correctly
        if image is None:
            print(f"Error reading image {image_file}")
            continue
        
        # Saves each annotated frame to the ./prediction_results/pred folder
        # classes = [1, 2, 3, 4] tells the model to leave out the swimbody annotation box
        results = model.predict(source=f"{image_file}", save=True, project="prediction_results", name="pred", classes = [1, 2, 3, 4], exist_ok=True, verbose=False)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"{len(image_files)} frames annotated in {duration} seconds")
    
    # Name of the file where the output video will be stored
    output_video_file = 'output_video.mp4'

    # Call the to_video function: compresses all the annotated frames back into an mp4 video file
    to_video(result_dir, output_video_file)

def main():
    # For testing purposes, I put an example input video in the project folder
    # Currently, the video is a multimedia container format (.MOV)
    # OpenCV should be able to process most video formats without changing the current code
    
    # Path to input video
    input_video_path = './example_video_input.MOV'
    
    # Make the call to the driver function
    predict(input_video_path)
    
main()

