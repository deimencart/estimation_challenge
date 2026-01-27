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

| Experiment | Sensor | Guess | Error Initial Guess | Error After 1st Meas. | Mean Error | Std Dev | Max Error | Final Error |
|------------|--------|-------|---------------------|----------------------|------------|---------|-----------|-------------|
| **Exp 1** | 3D | A (close) | 12.0 cm | **18.9 cm** | **21.0 cm** | 9.2 cm | 63.6 cm | 19.2 cm |
| **Exp 2** | 3D | B (far) | 173.5 cm | **19.0 cm** | **21.0 cm** | 9.2 cm | 63.6 cm | 19.2 cm |
| **Exp 3** | Camera | A (close) | 12.0 cm | **30.3 cm** | **102.9 cm**  | 65.2 cm | 347.7 cm | 2.0 cm |
| **Exp 4** | Camera | B (far) | 173.5 cm | **498.0 cm**  | **122.9 cm**  | 89.1 cm | 527.7 cm | 2.0 cm |

---

##  Detailed Analysis by Experiment

---

### **Experiment 1: 3D Sensor + Guess A** 
```
Sensor: 3D Position (σ = 14.14 cm)
Initial Guess: A (-0.7, -0.3, 6.6) - Distance: 12.0 cm
Filter: Classical Kalman Filter (Linear)
```

**Performance:**
- **Mean Error**: 21.0 cm
- **Final Error**: 19.2 cm
- **Convergence**: Immediate (1 timestep)

**Analysis:**
- Excellent baseline performance
- Error slightly higher than sensor noise (14 cm) due to model mismatch
- Model assumes constant velocity, but motion is circular (has acceleration)
- Very stable: low standard deviation (9.2 cm)

**Verdict:**  **EXCELLENT** - Works as expected

---

### **Experiment 2: 3D Sensor + Guess B** 
```
Sensor: 3D Position (σ = 14.14 cm)
Initial Guess: B (-1.8, -0.5, 7.9) - Distance: 173.5 cm
Filter: Classical Kalman Filter (Linear)
```

**Performance:**
- **Mean Error**: 21.0 cm (identical to Exp 1)
- **Final Error**: 19.2 cm (identical to Exp 1)
- **Convergence**: Ultra-fast (1-2 timesteps)

**Analysis:**
- **Initial guess effect**: Error drops from 173.5 cm → 19.0 cm in ONE measurement
- Filter "forgets" initial guess almost immediately
- Final performance identical to Exp 1
- Demonstrates **robustness** of KF to poor initialization

**Key Finding:**
> The 3D sensor provides enough information to correct even very poor initial guesses (14× worse than Guess A) within 1-2 timesteps.

**Verdict:** **EXCELLENT** - Very robust to initial guess

---

### **Experiment 3: Camera 2D + Guess A** 
```
Sensor: 2D Camera (σ = 2.24 px)
Initial Guess: A (-0.7, -0.3, 6.6) - Distance: 12.0 cm
Filter: Extended Kalman Filter (Non-linear)
```

**Performance:**
- **Mean Error**: 102.9 cm (5× worse than 3D sensor)
- **Max Error**: 347.7 cm (appears during convergence phase)
- **Final Error**: 2.0 cm (excellent after convergence)

**Analysis:**

**What went wrong:**
1. **Depth (Z) estimation problem**:
   - Camera measures (u, v) in pixels → does NOT measure Z directly
   - Z must be inferred from motion over time
   - Initial configuration causes Z to diverge temporarily

2. **Convergence pattern**:
   - Timesteps 0-50: High error (30-350 cm) as Z diverges
   - Timesteps 50-200: Gradual convergence
   - Timesteps 200+: Excellent performance (< 5 cm)

3. **Why final error is so good (2 cm)?**
   - Camera is MORE precise than 3D sensor (2.24 px vs 14 cm)
   - Once converged, it outperforms the 3D sensor
   - But convergence takes ~200 timesteps vs 1-2 for 3D sensor

**Configuration used:**
```python
P0 = diag([1.0, 1.0, 10.0, 1.0, 1.0, 1.0])
Q: q_std_xy = 0.1, q_std_z = 0.01
R = diag([5.0, 5.0])
```

**Verdict:****POOR MEAN, EXCELLENT FINAL** - Needs better tuning

---

### **Experiment 4: Camera 2D + Guess B** 
```
Sensor: 2D Camera (σ = 2.24 px)
Initial Guess: B (-1.8, -0.5, 7.9) - Distance: 173.5 cm
Filter: Extended Kalman Filter (Non-linear)
```

**Performance:**
- **Error after 1st measurement**: 498.0 cm (WORSE than initial guess!)
- **Mean Error**: 122.9 cm (6× worse than 3D sensor)
- **Max Error**: 527.7 cm
- **Final Error**: 2.0 cm (excellent after convergence)

**Analysis:**

**What went wrong:**
1. **Initial guess in Z is very wrong**: 7.9 m vs 6.6 m truth (1.3 m error)
2. **Camera cannot correct Z directly**:
   - First measurement provides (u, v) info
   - Filter tries to update (x, y, z) from 2D projection
   - Z correction is ambiguous → filter makes it WORSE initially
   
3. **Convergence is even slower**:
   - Takes ~300 timesteps to converge (vs 200 for Guess A)
   - High sensitivity to initial Z guess

**Key Finding:**
> Camera-based EKF is HIGHLY sensitive to initial guess, especially in Z coordinate. Poor initial guess can temporarily worsen the estimate before eventual convergence.

**Verdict:** ❌ **POOR** - Very sensitive to initial guess

---

## 📈 Comparative Analysis

### **1. How does the initial guess affect estimation w.r.t. each sensor?**

#### **3D Sensor:**
- **Almost no effect** on final performance
- Convergence in 1-2 timesteps regardless of guess quality
- Mean error: 21.0 cm (identical for Guess A and B)
- Final error: 19.2 cm (identical for Guess A and B)

**Conclusion:** The 3D sensor is **highly robust** to initial guess quality.

---

#### **Camera 2D:**
- **Significant effect** on convergence speed and mean error
- Guess A (close): Converges in ~200 timesteps
- Guess B (far): Converges in ~300 timesteps
- Mean error: 103 cm (Guess A) vs 123 cm (Guess B)
- Final error: 2.0 cm (identical for both, but takes longer)

**Conclusion:** The camera is **highly sensitive** to initial guess, especially in Z coordinate.

---

### **2. Pros and Cons of Each Sensor**

#### **3D Position Sensor**

| ✅ Pros | ❌ Cons |
|---------|---------|
| Measures X, Y, Z directly | High measurement noise (σ = 14 cm) |
| Simple linear measurement model (KF) | Noisier than camera |
| Very robust to initial guess | Cannot achieve sub-10cm accuracy |
| Fast convergence (1-2 timesteps) | - |
| Consistent performance | - |

**Best for:** Rapid initialization, robust tracking, real-time applications

---

#### **2D Camera**

| ✅ Pros | ❌ Cons |
|---------|---------|
| Very precise measurements (σ = 2.24 px) | Does NOT measure depth (Z) directly |
| Can achieve excellent final accuracy (2 cm) | Complex non-linear model (EKF required) |
| Lower measurement noise | Slow convergence (~200-300 timesteps) |
| - | Very sensitive to initial guess in Z |
| - | Requires careful parameter tuning (P0, Q) |
| - | Can diverge if misconfigured |

**Best for:** High-precision applications after convergence, static scenes

---

### **3. What could be done to improve it?**

#### **For 3D Sensor:**
1. **Better motion model**: Use circular motion model instead of constant velocity
2. **Sensor fusion**: Combine with camera for better precision
3. **Adaptive Q**: Adjust process noise based on estimated acceleration

#### **For Camera:**
1. 🔧 **Better initialization**:
   - Use 3D sensor for first few measurements to initialize Z
   - Implement multi-hypothesis tracking for Z
   
2. 🔧 **Improved process noise (Q)**:
   - Current: `q_z = 0.01` (too restrictive? or too loose?)
   - Need to find optimal balance
   - Consider adaptive Q based on estimation uncertainty
   
3. 🔧 **Better initial covariance (P0)**:
   - Current: `P0(Z) = 10.0` may be suboptimal
   - Consider: `P0(Z) = 20-50` for more flexibility
   
4. 🔧 **Sensor fusion**:
   - Combine camera + 3D sensor measurements
   - Use 3D sensor for Z, camera for XY precision
   
5. 🔧 **Stereo camera or depth camera**:
   - Would provide direct Z measurement
   - Eliminate depth ambiguity

---

## 🔧 Configuration Issues Found

### **Current Camera Configuration**
```python
P0 = np.diag([1.0, 1.0, 10.0, 1.0, 1.0, 1.0])
Q: q_std_xy = 0.1, q_std_z = 0.01
R = np.diag([5.0, 5.0])
```

### **Problems:**

1. **Mean error still high (103-123 cm)**:
   - Indicates temporary divergence during convergence
   - Z coordinate likely overshoots/undershoots initially
   
2. **Max error very large (348-528 cm)**:
   - Suggests filter goes significantly off-track mid-convergence
   - Needs better constraints on Z

### **Recommended Next Steps:**

Try these alternative configurations:

#### **Configuration A: More restrictive Z**
```python
P0 = np.diag([1.0, 1.0, 20.0, 1.0, 1.0, 1.0])  # More Z uncertainty
q_std_xy = 0.1
q_std_z = 0.005  # Even more restrictive
```

#### **Configuration B: More flexibility**
```python
P0 = np.diag([2.0, 2.0, 30.0, 2.0, 2.0, 2.0])
q_std_xy = 0.15
q_std_z = 0.01
```

---

## Final Performance Ranking

| Rank | Experiment | Mean Error | Convergence | Robustness |
|------|------------|------------|-------------|------------|
| 1st | **Exp 1: 3D + Guess A** | 21.0 cm | Immediate | Excellent |
| 2nd | **Exp 2: 3D + Guess B** | 21.0 cm | Immediate | Excellent |
| 3rd | **Exp 3: Camera + Guess A** | 102.9 cm | Slow (~200 ts) | Poor |
| 4th | **Exp 4: Camera + Guess B** | 122.9 cm | Very slow (~300 ts) | Very poor |

---

## Conclusions

### **Main Findings:**

1. **3D sensor** provides robust, fast, and consistent performance regardless of initial guess
2. **Camera** can achieve better final precision (2 cm) but suffers from:
   - Slow convergence
   - High sensitivity to initial guess
   - Temporary divergence during convergence
3. **EKF tuning is critical** for camera-based estimation
4. **Depth ambiguity** is the fundamental challenge with monocular cameras

### **Recommendations:**

**For practical applications:**
- Use **3D sensor** for initialization and rapid tracking
- Use **camera** for high-precision refinement after convergence
- Consider **sensor fusion** to get best of both worlds

**For improving camera performance:**
- Implement better initialization strategy
- Fine-tune Q matrix (especially `q_std_z`)
- Consider using stereo camera or depth sensor

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

