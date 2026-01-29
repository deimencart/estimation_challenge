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

### Solution 

#  Kalman Filter Experiments - Results Analysis

##  Executive Summary

Four experiments were conducted to estimate the 3D motion of a point moving in a circular trajectory using two sensor types (3D Position Sensor and 2D Camera) and two initial guesses (Guess A: close, Guess B: far).

**Key Findings:**
- **3D Sensor**: Excellent performance (~21 cm error), highly robust to initial guess
- **Camera 2D**: Poor performance (~103-123 cm error), highly sensitive to initial guess
- **EKF requires tuning**: Camera experiments show convergence issues due to depth (Z) estimation problems

---

## 🔬 Experimental Setup

### **Sensors**

| Sensor | Type | Measurement | Noise (σ) | Dimensionality |
|--------|------|-------------|-----------|----------------|
| **3D Position Sensor** | Direct | (x, y, z) in meters | 14.14 cm | 3D |
| **2D Camera** | Projection | (u, v) in pixels | 2.24 px | 2D |

### **Initial Guesses**

| Guess | Position [m] | Distance to Truth | Purpose |
|-------|--------------|-------------------|---------|
| **A** | (-0.7, -0.3, 6.6) | **0.12 m** | Test near convergence |
| **B** | (-1.8, -0.5, 7.9) | **1.74 m** | Test far convergence |

### **Ground Truth**

- **Motion**: Circular trajectory in 3D space
- **Plane**: XY plane (horizontal)
- **Radius**: ~2-3 meters
- **Height (Z)**: Constant at ~6.6 m
- **Duration**: 700 timesteps

---

##  Results Summary

### **Comparison Table**

| Experiment | Sensor | Guess | Error Initial Guess | Error After 1st Meas. | Mean Error | Std Dev | Max Error | Final Error| Model |
|------------|--------|-------|---------------------|----------------------|------------|---------|-----------|-------------|-------|
| **Exp 1** | 3D | A (close) | 12.0 cm | **18.9 cm** | **21.0 cm** | 9.2 cm | 63.6 cm | 19.2 cm |Velocity Const|
| **Exp 1b** | 3D | A (close) | 12.0 cm | **18.8 cm** | **16.9 cm** | 6.82 cm | 39.7 cm | 2.5 cm |Cons Position|
| **Exp 2** | 3D | B (far) | 173.5 cm | **19.0 cm** | **21.0 cm** | 9.2 cm | 63.6 cm | 19.2 cm |Velocity Const|
| **Exp 2b** | 3D | B (far) | 173.5 cm | **19.2 cm** | **16.9.0 cm** | 6.82 cm | 39.7 cm | 2.5 cm |Cons Position|
| **Exp 3** | Camera | A (close) | 12.0 cm | **30.3 cm** | **102.9 cm**  | 65.2 cm | 347.7 cm | 2.0 cm |Velocity Const|
| **Exp 4** | Camera | B (far) | 173.5 cm | **498.0 cm**  | **122.9 cm**  | 89.1 cm | 527.7 cm | 2.0 cm |Velocity Const|


## conlcusions 
1. For the 3d sensor experiments. the constant position model consstently outperfromed the constant velocity model. The linear model, assumes a linear motion wich is not matching the circular trajectory, this introduces systematic prediction errors. The constat position model, allow the measurments to domcunate the estimation process. 
   
**“How does the initial guess affect the estimation w.r.t. each sensor?**  
2. The experiments also showed that the initial guess has a limited long-term impact when using the 3D sensor. Despite large differences between the initial conditions (12 cm vs 173 cm), both configurations rapidly converged after the first measurement and reached nearly identical error statistics. This behavior indicates strong observability and reliable metric information provided by the 3D sensor.
    - 3D sensor: Experiments A and B goes more into th esolution, are almost the same
    - Camera : The guess B affects into the beginning, but in the end, goes into the solution 
   
3. camera experiments exhibited significantly higher mean errors, exceeding one meter in both initial guess scenarios. Although the filter eventually converged to small final errors, large transient errors dominated the trajectory. This behavior reflects the inherent geometric limitations of monocular vision while lateral position (X, Y) is well constrained through image measurements, depth (Z) remains weakly observable.
   - 3D sensor: stable, however a deeper study on R and Q may be necessary. 
   - weak depth estimation, maybe an stereo system may work better, a fusion in between sensors may work better. 

---

## Files Generated
```
results_1/
├── exp1_3d_sensor_guess_a_states.npy
├── exp1_3d_sensor_guess_a_errors.npy
└── exp1_3d_sensor_guess_a.png

results_2/
├── exp2_3d_sensor_guess_b_states.npy
├── exp2_3d_sensor_guess_b_errors.npy
└── exp2_3d_sensor_guess_b.png

results_3/
├── exp3_camera_guess_a_states.npy
├── exp3_camera_guess_a_errors.npy
└── exp3_camera_guess_a.png

results_4/
├── exp4_camera_guess_b_states.npy
├── exp4_camera_guess_b_errors.npy
└── exp4_camera_guess_b.png
```

---

## Technical Notes

### **Kalman Filter Configuration (Experiments 1-2)**
```python
Model: Constant velocity
F: 6×6 state transition matrix
Q: Process noise (q_std = 0.5)
H: 3×6 measurement matrix (direct)
R: diag([0.02, 0.02, 0.02]) m²
P0: eye(6) * 1.0
```

### **Extended Kalman Filter Configuration (Experiments 3-4)**
```python
Model: Constant velocity + non-linear projection
F: 6×6 state transition matrix
Q: Non-uniform (q_xy = 0.1, q_z = 0.01)
h(x): Pinhole camera projection (non-linear)
H(x): Jacobian (computed dynamically)
R: diag([5.0, 5.0]) px²
P0: diag([1.0, 1.0, 10.0, 1.0, 1.0, 1.0])
```

