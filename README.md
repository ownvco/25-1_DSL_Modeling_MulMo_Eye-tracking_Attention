# LOOKAT: Reading disability feedback system based on eye-tracking, reading-disability detection, OCR, and Large Language model

## Statement
This code was produced for the modeling project in Data Science Lab@Yonsei, February - March, 2025 by multi-modal team. 

Minkyung Baek, Jinny Seo, Jaehwa Yang, Jisung Jo, and Jimin Bok contributed to this project, and below is the instruction about running the program and the architecture of packages.


## Introduction
Our model LOOKAT aims to provide feedbacks to the children having reading disabilities, based on the data collected from the eye-tracking
model, reading disability detection model, OCR model, and Large Language Model.

This pipeline aims to implement every single step into end-to-end model.

The data streams are transmitted internally through python queue package, and the user could verify the procedure by checking out the visualized data, which are located in ./inputdata, ./rawdata, ./screenshots. 

If you want to test our program on your local environment, there are some requirements that should be met.

## Requirements

### Dependencies

The codes were fully generated in python language, and following packages are required.

Note that you'll definitely need to have two virtual environments (e.g. conda environments), as some of the python packages used to run the end-to-end model has some conflicts. (Specifically, mediapipe and paddleocr has some confilcts on their dependencies)

I highly recommend to download the paddleocr dependencies on the main environment, and download the mediapipe dependencies on the sub-environement.

After you've downloaded all the dependencies on your environments, you should modify the hard-coded environment directory to something that is your own. 

in Line 266 on Control_new.py, you should modify the line to
```python
    jay_python = r"path\to\your\environment"
```

Not only that, you should type your own api key in the ./LLM/feedback.py

Dependencies (main environment)

    python = 3.9.19
    torch = 1.13.0
    torchvision = 0.14.0
    pyautogui = 0.9.54
    opencv = 4.10.9
    openai = 1.68.2
    matplotlib = 3.7.1
    watchdog = 3.0.0
    scikit-learn = 1.2.1
    dlib = 19.24.6
    pillow = 10.2.0
    paddlepaddle = 2.6.2
    paddleocr = 2.10.0
    pandas = 2.2.3
    numpy = 1.26.4

Dependencies (subenvironment, where you will run the gaze tracking and calibration program)

    python = 3.12.8
    mediapipe = 0.10.21
    scikit-learn = 1.6.1
    pillow = 11.1.0
    numpy = 1.26.4

Dependencies could be updated afterwards.
    
### Directories and calling the python script for implementing

Of course, you'll be running the whole code on the non-GPU compatible environment, i.e. laptops. Therefore, I believe most of the users would have webcams equipped already.

If not, I assume that this code would not work. Also in the case you hav auxiliary webcams, I'm not sure our code would work, therefore you might modify the codes designating which webcams to use from the ./new_tracking_api/realtime_gaze_tracking.py and one_point_calibration.py. (maybe from line 63 and 69, respectively)

wherever you place the root folder (default: Integrated), you should move your current working directory to the root directory, and type the line below to the command

```powershell
python Control_new.py
```
If you run the script, you will get an prompt asking whether you want initiation or not. 
This initiation will delete all of the files included in ./rawdata, ./inputdata, ./Calibration_new, ./screenshots,...
which are the directories where the data will be stored while running single trial.

```powershell
Sure that you want to run initialization?
([Y]/N):Traceback (most recent call last):
```
You can Just press Enter or type y into the line, to run the program.

After initialization, you'll be asked to calibrate the camera. Once you get the black display with the huge red dot on the middle, please focus on that red dot, and press enter if you think that you have given sufficient attention.

Converting your window to where your script is located in, the recording will start after the light on your webcam, indicating that recording is in progress, has been lit.

You can enjoy your reading, and the feedback based on your reading behavior will come up on the cmd line. Also, the 30 second-wise classification result will also be depicted, tandemmly with the real-time feedback.

Empirically, this model works better when run on the cmd, rather than vscode powershell environment :)

Integrated

|-Control (Controlling the packages encapsulated in the system, but currently **Depracted**)

|-Control_new (Controlling the packages, based on the newly developed eye-tracking system. Transmits the data stream with python queue datatype)

|-README.md

|-Init.py (Utilized for initializing the data being stored)

|-new_tracking_api (Provides the packages to run the calibration, eye-tracking, and rule based reading disability detection)

    |- one_point_calibration.py (For running the calibration)

    |- realtime_gaze_tracking.py (For running the real-time 
    gaze tracking)

    |- detection.py (Detects the fixation, regression from the real-time data stream)

    |- processor.py (Detects where the fixation elongation took place)

|- dataprocessor (Where the .csv type 30s-wise data are processed into input image of our DL based classifier model)
    |- trace_processing (The exact module which processes the .csv file into image data)

|- Calibration_new (Where calibration data are stored. Wiped out if the initilization process is undergone)

|- classifier (Contains the pretrained weights and classifying model)

    |- DDD_efficientnet.py (Contains the exact model)
    |- pretrained_weights (Contains the pretrained weights)

|- LLM (Contains feedback LLM model)

|- inputdata (Contains the trace map that are harnessed as input data of classification model)

|- rawdata (.csv files generated every 30 seconds, which will be converted to inputdata)

|- screenshots (Where screenshots from fixation elongation and regression are stored)

|- OCR (Where OCR module and screenshot capturing modules are stored)


### To verify if your model is working properly

I've provided some directories where you can find the recorded data within the pipeline of conveying the feedback to our user.

Check out the files inside the ./screenshots, ./inputdata. This will tell you whether the model captures your gaze properly and whether the classification is done decently.

Now, you can go and give it a try to run our LOOKAT model!
