## Programming challenge
Your task is to estimate the motion of a 3D point in space over time. The point is performing a circular motion which is observed using sensors.

### Setup
You have two sensors available, which you will use individually:
a) the position sensor is located at (0,0,0) and measures the 3D position [m] of a point. But the sensor is quite noisy. Its measurement noise is modeled as covariance matrix with diagonal(0.02,0.02,0.02)[m^2]
b) the camera sensor is also located at (0,0,0). It measures the projected point on the camera image as a 2D measurement. Its measurement noise is better and is modeled as covariance matrix with diagonal(5,5)[pixel^2]. The camera follows a simplified pinhole model where its focal length is f=500 pixels, image_width=640 pixels, image_height=480 pixels.

You can start your estimation from two different initial guesses:
A: [-0.7, -0.3, 6.6]
B: [-1.8,-0.5,7.9]


### Task
your task is to:
- run the estimation at least in these 4 variations: [position sensor / camera sensor] x [ initial guess A / initial guess B]. 
- estimate the circle that the point moves in space
- visualize the 3D motion in all cases together with the ground truth
- compare your estimation against the ground truth and visualize the error
- How does the initial guess affect the estimation w.r.t. each sensor?
- what are the pros and cons for each of the sensor types? What could you do to improve it?
- upload your code (preferably python, but could also be c++, matlab, ...), your plots, and your discussion to github.

Recommendation: use the classical Kalman Filter algorithm to solve this task. For the camera part, you will need to linearize, thus using the "Extended Kalman Filter" approach.


### Input
I will provide you with 3 files:
- `measurements_2d_camera.csv` the 2D camera observations
- `measurements_3d_camera.csv` the 3D point sensor observations
- `ground_truth_3d_position.csv` the 3D reference ground truth

Note: each line corresponds to a specific time. I.e. line X of each file correspond to the same timestamp.