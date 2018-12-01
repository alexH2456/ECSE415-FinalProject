=============================================================================
This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 
4.0 International License. To view a copy of this license, visit 

http://creativecommons.org/licenses/by-nc-sa/4.0/ 

or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
=============================================================================

The 2017 MIO-TCD localization dataset contains 137,743 high resolution images 
each containing one or more foreground objects.  These objects are classified
as follows:

	articulated_truck
	bicycle
	bus
	car
	motorcycle
	motorized_vehicle	
	non-motorized_vehicle
	pedestrian
	pickup_truck
	single_unit_truck
	work_van
	
The motorized_vehicle class is for vehicles that cannot be labeled into a specific category 
(this can be due to occlusion or perpective).

The dataset contains 110,000 training images (~80%) and 27,743 testing images (~20%)

The goal of this challenge is to **correctly localize and label each vehicle**.  The 
output of your method shall to be put in a csv format such as 'gt_train.csv',
'your_results_test.csv', or 'your_results_train.csv' provided with the dataset.
'gt_train.csv' contains ground truth while 'your_results_test.csv' and
'your_results_train.csv' contains a random class assignment to the training and
testing images.

NOTE: Python code is available online to help you play around with our dataset:
    tcd.miovision.com/challenge/dataset/

you can visualize the images with the bounding boxes on it with the following command

> python view_bounding_boxes.py ./train/ ./gt_train.csv 


You may run the following command to parse the dataset and produce a valide
csv file :

> python parse_localization_dataset.py ./train/ your_results_train.csv
or
> python parse_localization_dataset.py ./test/ your_results_test.csv

You may also measure your training score with the 'localization_evaluation.py'
python code. For this, you only need to run the following command in a terminal

> python localization_evaluation.py gt_train.csv your_results_train.csv

That code was developed and tested with python 3.5.2 on Linux.

