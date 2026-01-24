# Análisis Visual de los Datos - Kalman Filter Challenge
# Data Visualization Analysis

### Figure 1: Three-View Data Overview

<img width="5258" height="1775" alt="data_visualization" src="https://github.com/user-attachments/assets/1748bba6-a968-42b2-890e-4976c0d76c35" />

This figure presents three complementary views of the tracking problem, showing the ground truth trajectory, sensor measurements, and initial guess locations.

---

## Plot 1: 3D Trajectory View

This plot shows the complete 3D motion of the point along with sensor measurements and initial conditions.

### Ground Truth (Blue Line)
- Represents the actual position of the point over time
- Forms a perfect circle in 3D space
- Constant height: Z = 6.6 meters
- Approximate radius: 2.0 meters
- Approximate center: (1.35, -0.24, 6.6) meters
- The point completes multiple full rotations around the center

### 3D Position Sensor Measurements (Red Points)
- Direct measurements of 3D coordinates (x, y, z)
- High noise level: σ ≈ 0.14 m (standard deviation)
- Points are dispersed around the true trajectory
- Displayed subsampled (every 10th point) for visual clarity
- Measurement noise covariance: R = diag(0.02, 0.02, 0.02) m²

### Sensor Location (Yellow Star)
- Position: (0, 0, 0)
- Represents the location of BOTH sensors:
  - 3D position sensor
  - 2D camera sensor
- Both sensors are fixed at the origin
- The point moves at approximately 7 meters distance from the sensors

### Initial Point (Green Circle)
- Marks the initial position of the point at t=0
- Approximate position: (-0.70, -0.18, 6.6) meters

### Final Point (Purple Square)
- Marks the final position of the point at t=699
- After completing multiple circular rotations

### Initial Guess A (Orange Triangle)
- First proposed initial estimate
- Position: (-0.7, -0.3, 6.6) meters
- Very close to the actual initial point
- Distance to real point: ≈ 0.12 meters
- Expected to result in rapid filter convergence

### Initial Guess B (Purple Triangle)
- Second proposed initial estimate
- Position: (-1.8, -0.5, 7.9) meters
- Far from the actual initial point
- Distance to real point: ≈ 1.74 meters (14× farther than Guess A)
- Expected to result in slow convergence or potential difficulties

### Key Observations
1. The motion is perfectly circular in the XY plane
2. The Z coordinate remains constant (2D motion in a horizontal plane)
3. The 3D sensor noise is considerable and visually apparent
4. The two initial guesses have significantly different distances to the true initial state

---
