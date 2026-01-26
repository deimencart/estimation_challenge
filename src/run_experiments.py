"""
run_experiments.py

Main script to execute the 4 Kalman Filter experiments:
1. 3D Sensor + Guess A
2. 3D Sensor + Guess B
3. Camera 2D + Guess A
4. Camera 2D + Guess B
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Import Kalman Filter classes
from kalman_filter import (
    KalmanFilter,
    ExtendedKalmanFilter,
    create_motion_model,
    camera_projection,
    camera_jacobian
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Initial guesses
GUESS_A = np.array([-0.7, -0.3, 6.6])
GUESS_B = np.array([-1.8, -0.5, 7.9])

# Camera parameters
FOCAL_LENGTH = 500  # pixels
CX = 320  # image center x
CY = 240  # image center y

# Time step (assuming normalized time)
DT = 1.0

# Process noise standard deviation
Q_STD = 0.5


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """
    Load measurement data from CSV files.
    
    Returns:
    --------
    ground_truth : ndarray (n, 3)
        True 3D positions
    sensor_3d : ndarray (n, 3)
        3D sensor measurements
    camera_2d : ndarray (n, 2)
        Camera measurements in pixels
    """
    file_path = "D:/Dev_Space/estimation_challenge/"
    ground_truth_file = file_path + "resources/ground_truth_3d_position.csv"
    camera_2d_file = file_path + "resources/measurements_2d_camera.csv"
    sensor_3d_file = file_path + "resources/measurements_3d_position_sensor.csv"

    ground_truth = np.loadtxt(ground_truth_file, delimiter=',', skiprows=1)
    sensor_3d = np.loadtxt(sensor_3d_file, delimiter=',', skiprows=1)
    camera_2d = np.loadtxt(camera_2d_file, delimiter=',', skiprows=1)
    
    print("Ground Truth Shape:", ground_truth.shape)
    print("3D Sensor Measurements Shape:", sensor_3d.shape)
    print("2D Camera Measurements Shape:", camera_2d.shape)
    print()
    
    return ground_truth, sensor_3d, camera_2d


# ============================================================================
# EXPERIMENTS
# ============================================================================

def run_experiment_3d_sensor(measurements, initial_guess, experiment_name):
    """
    Run Kalman Filter with 3D position sensor.
    
    Parameters:
    -----------
    measurements : ndarray (n, 3)
        3D sensor measurements
    initial_guess : ndarray (3,)
        Initial position estimate [x, y, z]
    experiment_name : str
        Name for logging
    
    Returns:
    --------
    estimated_states : ndarray (n, 6)
        Estimated states over time
    """
    print(f"Running {experiment_name}...")
    
    # Create motion model
    F, Q = create_motion_model(DT, q_std=Q_STD)
    
    # Initial state (position + zero velocity)
    x0 = np.concatenate([initial_guess, np.zeros(3)])
    
    # Initial covariance
    P0 = np.eye(6) * 1.0
    
    # Measurement matrix for 3D sensor (measures x, y, z directly)
    H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ])
    
    # Measurement noise covariance
    R = np.diag([0.02, 0.02, 0.02])
    
    # Create Kalman Filter
    kf = KalmanFilter(x0, P0, F, Q, H, R)
    
    # Run filter through all measurements
    n_measurements = len(measurements)
    for i, z in enumerate(measurements):
        # Predict
        kf.predict()
        
        # Update with measurement
        kf.update(z)
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_measurements} measurements")
    
    # Get estimated trajectory
    estimated_states = kf.get_state_history()
    
    print(f" {experiment_name} complete!")
    print()
    
    return estimated_states


def run_experiment_camera(measurements, initial_guess, experiment_name):
    """
    Run Extended Kalman Filter with camera sensor.
    
    Parameters:
    -----------
    measurements : ndarray (n, 2)
        Camera measurements in pixels [u, v]
    initial_guess : ndarray (3,)
        Initial position estimate [x, y, z]
    experiment_name : str
        Name for logging
    
    Returns:
    --------
    estimated_states : ndarray (n, 6)
        Estimated states over time
    """
    print(f"Running {experiment_name}...")
    
    # Create motion model
    F, Q = create_motion_model(DT, q_std=Q_STD)
    
    # Initial state (position + zero velocity)
    x0 = np.concatenate([initial_guess, np.zeros(3)])
    
    # Initial covariance - higher for camera (loses depth info)
    P0 = np.diag([10.0, 10.0, 50.0, 5.0, 5.0, 5.0])
    
    # Measurement noise covariance (camera)
    R = np.diag([5.0, 5.0])
    
    # Create measurement function and Jacobian function
    def h_func(x):
        return camera_projection(x, FOCAL_LENGTH, CX, CY)
    
    def jacobian_func(x):
        return camera_jacobian(x, FOCAL_LENGTH)
    
    # Create Extended Kalman Filter
    ekf = ExtendedKalmanFilter(x0, P0, F, Q, R, h_func, jacobian_func)
    
    # DEBUG: First iteration
    print("\n*** DEBUG: First Iteration ***")
    ekf.predict()
    print(f"  x_pred[0:3]: {ekf.x_pred[0:3]}")
    
    z_pred = h_func(ekf.x_pred)
    print(f"  z[0]: {measurements[0]}")
    print(f"  z_pred: {z_pred}")
    print(f"  Innovation: {measurements[0] - z_pred}")
    
    ekf.update(measurements[0])
    print(f"  x after update[0:3]: {ekf.x[0:3]}")
    print("*** End Debug ***\n")
    
    # Run filter through remaining measurements
    n_measurements = len(measurements)
    for i in range(1, n_measurements):  # Start from 1 (already did 0)
        z = measurements[i]
        
        # Predict
        ekf.predict()
        
        # Update with measurement
        ekf.update(z)
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_measurements} measurements")
    
    # Get estimated trajectory
    estimated_states = ekf.get_state_history()
    
    print(f"  ✓ {experiment_name} complete!")
    print(f"  Final trajectory shape: {estimated_states.shape}")
    print()
    
    return estimated_states


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def compute_errors(estimated_states, ground_truth):
    """
    Compute position errors between estimated and true trajectories.
    
    Parameters:
    -----------
    estimated_states : ndarray (n, 6)
        Estimated states [x, y, z, vx, vy, vz]
    ground_truth : ndarray (n, 3)
        True positions [x, y, z]
    
    Returns:
    --------
    position_errors : ndarray (n,)
        Euclidean distance errors at each timestep
    """
    # Extract positions from states
    estimated_positions = estimated_states[:, 0:3]
    
    # Compute Euclidean distance errors
    errors = np.linalg.norm(estimated_positions - ground_truth, axis=1)
    
    return errors


def print_statistics(errors, experiment_name):
    """Print error statistics."""
    print(f"Statistics for {experiment_name}:")
    print(f"  Mean error: {errors.mean():.4f} m")
    print(f"  Std error:  {errors.std():.4f} m")
    print(f"  Max error:  {errors.max():.4f} m")
    print(f"  Final error: {errors[-1]:.4f} m")
    print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print("="*70)
    print("KALMAN FILTER EXPERIMENTS")
    print("="*70)
    print()
    
    # Load data
    ground_truth, sensor_3d, camera_2d = load_data()
    
    # ========================================================================
    # EXPERIMENT 1: 3D Sensor + Guess A
    # ========================================================================
    states_exp1 = run_experiment_3d_sensor(
        sensor_3d, 
        GUESS_A, 
        "Experiment 1: 3D Sensor + Guess A"
    )
    errors_exp1 = compute_errors(states_exp1, ground_truth)
    print_statistics(errors_exp1, "Experiment 1")
    
    # ========================================================================
    # EXPERIMENT 2: 3D Sensor + Guess B
    # ========================================================================
    states_exp2 = run_experiment_3d_sensor(
        sensor_3d, 
        GUESS_B, 
        "Experiment 2: 3D Sensor + Guess B"
    )
    errors_exp2 = compute_errors(states_exp2, ground_truth)
    print_statistics(errors_exp2, "Experiment 2")
    
    # ========================================================================
    # EXPERIMENT 3: Camera + Guess A
    # ========================================================================
    states_exp3 = run_experiment_camera(
        camera_2d, 
        GUESS_A, 
        "Experiment 3: Camera 2D + Guess A"
    )
    
    # DEBUG: Check shapes and first errors
    print("\n*** DEBUG Experiment 3 ***")
    print(f"states_exp3 shape: {states_exp3.shape}")
    print(f"ground_truth shape: {ground_truth.shape}")
    print(f"First 5 positions estimated:")
    for i in range(5):
        print(f"  [{i}] {states_exp3[i, 0:3]}")
    print(f"First 5 positions ground truth:")
    for i in range(5):
        print(f"  [{i}] {ground_truth[i]}")
    print(f"First 5 errors:")
    for i in range(5):
        err = np.linalg.norm(states_exp3[i, 0:3] - ground_truth[i])
        print(f"  [{i}] {err:.4f} m")
    print("*** End Debug ***\n")
    
    errors_exp3 = compute_errors(states_exp3, ground_truth)
    print_statistics(errors_exp3, "Experiment 3")
    
    # ========================================================================
    # EXPERIMENT 4: Camera + Guess B
    # ========================================================================
    states_exp4 = run_experiment_camera(
        camera_2d, 
        GUESS_B, 
        "Experiment 4: Camera 2D + Guess B"
    )
    errors_exp4 = compute_errors(states_exp4, ground_truth)
    print_statistics(errors_exp4, "Experiment 4")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    results_dir = Path("D:/Dev_Space/estimation_challenge/results_dir/")
    results_dir.mkdir(exist_ok=True)
    
    np.save(results_dir / 'exp1_states.npy', states_exp1)
    np.save(results_dir / 'exp2_states.npy', states_exp2)
    np.save(results_dir / 'exp3_states.npy', states_exp3)
    np.save(results_dir / 'exp4_states.npy', states_exp4)
    
    np.save(results_dir / 'exp1_errors.npy', errors_exp1)
    np.save(results_dir / 'exp2_errors.npy', errors_exp2)
    np.save(results_dir / 'exp3_errors.npy', errors_exp3)
    np.save(results_dir / 'exp4_errors.npy', errors_exp4)
    
    print("✓ Results saved to:", results_dir)
    print()
    print("="*70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()