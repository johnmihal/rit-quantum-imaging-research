# Meep Simple Star Coupler Simulation Repository

## Table of Contents
1. [Introduction to the Repository](#1)
2. [Running the Code](#2)
   1. [Installing Meep](#2.1)
   2. [Known Meep Issues](#2.2)
3. [Where I Left Off](#3)

## Introduction to the Repository <a name="1"></a>
The repository is split up into multiple folders. 

The working and most recent code is not in any folder, but is instead at this level. This includes `metrics.py`, `Simple_Star_Coupler_SM_Experiment_wArgs.py`, `test.py`, `transform_matrix.py`, and `transform_matric.ipynb`. `test.py` is where I test random snippets of python code, it has not relation to the simulation and tranformation matricies. 

The `example code` folder stores files from the meep website which I used for development. These files have code with many usfull exmaples on how to do things

The `old code` folder conatins previous versions of the star coupler simuation code. 

The `outputs` folder conatins all the outputs that the code produces. There are alot of outputs so putting them there declutters things,

## Running the Code <a name="1"></a>
To run the main experiment activte the meep conda enviornment. The run the following command `./Simple_Star_Coupler_SM_Experiment_wArgs <first> <last> (--s|--sv|--lv|--l) [--r=<um>] [--n=<>]`

Useage:
```
Mandatory Values:
    first  First waveguide to test, inclusive.
    last   Last waveguide to test, inclusive.
    --s
    --sv    Short test mode, with video.
    --lv   Long simulation, with video. 
    --l    Long simulation, no video.    
Options:
    --r=<um>  Star coupler radius (um) [default: 30].
    --n=<>    Number of waveguides on each side [default: 21].
```

Note: Running the simulation with video is much slower. I found that at most of the time I ran in --lv (long video mode) the simluation would crash because making the video and normalizing all of the data took up too much space and absolutly killed the ram.

The simulation will create folders with names such as `Simple_Star_coupler_SM_DATA_0`. This indicates that the folder contains data from testing the input on wave guide 0. The last number denotes the input waveguide for the test run. 

Within the folder you will see a fields.h5 and structure.h5 folder. Those can be loaded back into meep to reload the simluation in. **Unfortunately meep can not load in monitors from dumps.** Because of this the simulation pickles the data from each monitor. These pickles are in the files named like `itr_0_input_1`. The 0 means this monitor was iteration 0. This will be the same as the folder number. The 1 indicates which waveguide the monitor was on. And the input, or output tells use weather this was from an input or output waveguide monitor.

The `transfer_matrix.py` uses this naming convention to load in the pickled data to create the transfer matrix. 

From there the tranfermatrix could be analyzed using the function in the `metrics.py` file. However at this time none of this is set up. 

One nice thing to do when running the simulation is to run the code as such:
`python Simple_Star_Coupler_SM_Experiment_wArgs.py 0 0 --l --r=35 --n=8 2>&1 | tee terminal.txt`
The `2>&1 | tee terminal.txt` function puts all of the terminal output and errors into a file called terminal.txt, this is really helpful for debugging. If the simulation fails like 5 hours in having a record of what happened makes debugging much easier.

Another thing to notes is that Meep doesn't do gpu acceleration that I am aware of. I actually found that this ran faster on my M1 mac than on my gaming pc which had a meh cpu and a good gpu. It was much much faster on my mac.


### Installing Meep <a name="2.1"></a>
Meep is most easily installed using conda. Go to the meep website for information about this. Simply use the recomended conda install, all the other stuff is too complicated. For MacOS and Linux the install is simple, use recommended conda install. For windows install the WSL(Windows Subsystem for Linux) and install Meep onto the WSL as if it was a Mac or Linux machine.

### Known Meep Issues <a name="2.2"></a>
If your meep installaion keeps crashing when running the code and complains about a GSL error then it is because the GSL that comes with meep is broken. You will need to install the a compadible and not broken GSL version into your conda meep installation. Simply find the newest and compadible GSL that works with your meep stuff. 

## Where I Left Off <a name="3"></a>
The code all works! Simply use `Simple_Star_Coupler_SM_Experiment_wArgs.py` generate the data. Then load the data and analyze the data using 'evaluation_main.py'.  'evaluation_main.py' uses `transfer_martix.py` to load the data and `metrics.py` to analyze it. Evan gave me `metrics.py` so he could compare my data with his. 

I never got to compare the end results with Evan because at the very end the bottom fell out of the code and it stopped working, BUT I fixed it all!

Evan wanted me to generate more data. I only generated a radius of 35 and 8 by 8 star coupler simluation. It is linked here: https://drive.google.com/file/d/1Pbmff1Da90Qc_qKwCkOzP1R9gC8wiwY9/view?usp=sharing

Download the data to test the evaluation. I will say as I haven't compared the evaluation with Evan or anyone I am not a 100% sure the evaluation does what it is supposed to do, but it doesn't crash so thats a win for me.

But yeah it should all be working!