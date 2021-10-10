This is course project of [ECE276A(UCSD)](https://natanaso.github.io/ece276a/). Thanks for the instructor and TAs for providing the test dataset and help.
### [Formal report](./SLAM_KL.pdf)
* Problem formulation
* Algorithm description
* Test results on three datasets

### Sensor on the vehicle
* IMU (100Hz): data/*.npz, angular/linear velocity of the car
* Camera (20Hz): data/*.avi, video taken from the car with object-tracking results

### Scripts: 
* utils.py: 
	* contains all supporting functions
* hw3_main.py:
	* Variables to be changed: 
		* ID: dataset
		* n1: noise level of motion model
		* n2: noise level of observation model
	* Plots to be generated: (refer to the report for more details)
		* Prediction*.png: prediction-only pose trajectoru
		* Initial_Map*.png: map initialized using prediction-only pose
		* MapUpdate*.png: visual mapping of features
		* FullSLAM*.png: visual inervial SLAM based on prediction-baised map initialization
		* FullSLAMv2*.png: visual inervial SLAM with constrained update 
