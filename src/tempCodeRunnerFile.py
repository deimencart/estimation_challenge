"""
run_experiments.py

Main script to execute the 4 Kalman Filter experiments:
1. 3D Sensor + Guess A
2. 3D Sensor + Guess B
3. Camera 2D + Guess A
4. Camera 2D + Guess B
"""

from email import errors
from xml.parsers.expat import errors
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Import Kalman Filter classes
from kalman_filter import (
    KalmanFilter,
    ExtendedKalmanFilter,
    create_motion_model,
    camera_projection,
    camera_jacobian, 
    constant_position_model
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


# DATA LOADING


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


def experiment_a_consPos(measurments, ground_truth):
    """
    Experiment 1: 3D Sensor + Guess A with Constant Position Model

    Parameters: 
    messurments: ndarray (700, 3)
        3D sensor measurements
    ground_truth: ndarray (700, 3)

    
    Returns:
    estimated_states: ndarray (700, 6)
        Estimated states over time
    errors: ndarray (700, )
        Position estimation errors over time
    """

    X0 = GUESS_A  # Initial state with zero velocity
    P0 = np.eye(3) * 1.0  # Initial covariance

    # Para posición constante, q_std es el desplazamiento esperado por timestep
    # Movimiento circular: v ≈ ω × r ≈ 0.063 m/s
    # Desplazamiento por timestep ≈ v × dt ≈ 0.063 m
    F, Q = constant_position_model(DT, q_std=0.120)


    #Sensor model H for 3D position sensor

    H = np.eye(3)
    #This is the direct MEasurment of the postion 

    R = np.diag([0.02, 0.02, 0.02])  # Measurement noise covariance given by the challenge
    print(f"  Ruido del sensor: σ = {np.sqrt(np.diag(R)[0]):.4f} m (≈14 cm)")
    print()

    # Creating Kalman Filter instance

    kf = KalmanFilter(X0, P0, F, Q, H, R)
    print("Creating Kalman Filter")
    print()

    n_mesurments = len(measurments)
    print ("Starting Experiment 1b: 3D Sensor + Guess A with Constant Position Model")
    for i in range(n_mesurments):
        z = measurments[i]
        kf.predict()
        kf.update(z)

        # Showing progress
        if (i + 1) % 100 == 0 or i == n_mesurments - 1:
            print(f"  Time Step {i + 1}/{n_mesurments} completed.")
    print ("Filter completed.")
    print ()

    estimated_states = kf.get_state_history()
    errors = []
    for i in range(len(estimated_states)):
        estimated_pos = estimated_states[i]  # Ya es 3D, no necesita slicing
        true_pos = ground_truth[i]
        error = np.linalg.norm(estimated_pos - true_pos)
        errors.append(error)
    
    errors = np.array(errors)
    error_init = np.linalg.norm(GUESS_A - ground_truth[0])
    print("Results - EXPERIMENT 1b - Constant Position Model")
    print("="*70)
    print(f"Error initial guess: {error_init:.4f} m  ({error_init*100:.1f} cm)")
    print(f"Error After first measurement:   {errors[0]:.4f} m  ({errors[0]*100:.1f} cm)")
    print(f"Mean Error:          {errors.mean():.4f} m  ({errors.mean()*100:.1f} cm)")
    print(f"Standard Deviation:     {errors.std():.4f} m")
    print(f"Error max:            {errors.max():.4f} m")
    print(f"Error min:            {errors.min():.4f} m")
    print(f"Error final:             {errors[-1]:.4f} m  ({errors[-1]*100:.1f} cm)")
    print()
    
    return estimated_states, errors

def experiment_b_consPos(measurments, ground_truth):
    """
    Experiment 2b: 3D Sensor + Guess B with Constant Position Model

    Parameters: 
    messurments: ndarray (700, 3)
        3D sensor measurements
    ground_truth: ndarray (700, 3)

    
    Returns:
    estimated_states: ndarray (700, 6)
        Estimated states over time
    errors: ndarray (700, )
        Position estimation errors over time
    """

    X0 = GUESS_B  # Initial state with zero velocity
    P0 = np.eye(3) * 1.0  # Initial covariance

    # Para posición constante, q_std es el desplazamiento esperado por timestep
    # Movimiento circular: v ≈ ω × r ≈ 0.063 m/s
    # Desplazamiento por timestep ≈ v × dt ≈ 0.063 m
    F, Q = constant_position_model(DT, q_std=0.120)


    #Sensor model H for 3D position sensor

    H = np.eye(3)
    #This is the direct MEasurment of the postion 

    R = np.diag([0.02, 0.02, 0.02])  # Measurement noise covariance given by the challenge
    print(f"  Ruido del sensor: σ = {np.sqrt(np.diag(R)[0]):.4f} m (≈14 cm)")
    print()

    # Creating Kalman Filter instance

    kf = KalmanFilter(X0, P0, F, Q, H, R)
    print("Creating Kalman Filter")
    print()

    n_mesurments = len(measurments)
    print ("Starting Experiment 2b: 3D Sensor + Guess B with Constant Position Model")
    for i in range(n_mesurments):
        z = measurments[i]
        kf.predict()
        kf.update(z)

        # Showing progress
        if (i + 1) % 100 == 0 or i == n_mesurments - 1:
            print(f"  Time Step {i + 1}/{n_mesurments} completed.")
    print ("Filter completed.")
    print ()

    estimated_states = kf.get_state_history()
    errors = []
    for i in range(len(estimated_states)):
        estimated_pos = estimated_states[i]  # Ya es 3D, no necesita slicing
        true_pos = ground_truth[i]
        error = np.linalg.norm(estimated_pos - true_pos)
        errors.append(error)
    
    errors = np.array(errors)
    error_init = np.linalg.norm(GUESS_B - ground_truth[0])
    print("Results - EXPERIMENT 2b")
    print("="*70)
    print(f"Error initial guess: {error_init:.4f} m  ({error_init*100:.1f} cm)")
    print(f"Error After first measurement:   {errors[0]:.4f} m  ({errors[0]*100:.1f} cm)")
    print(f"Mean Error:          {errors.mean():.4f} m  ({errors.mean()*100:.1f} cm)")
    print(f"Standard Deviation:     {errors.std():.4f} m")
    print(f"Error max:            {errors.max():.4f} m")
    print(f"Error min:            {errors.min():.4f} m")
    print(f"Error final:             {errors[-1]:.4f} m  ({errors[-1]*100:.1f} cm)")
    print()
    return estimated_states, errors

def experiment_c_consPos(measurments, ground_truth):
    """
    Experimento: Extended Kalman Filter con Cámara 2D + Modelo de Posición Constante
    
    Parameters:
    -----------
    measurements : ndarray (700, 2)
        Camera measurements [u, v] in pixels
    ground_truth : ndarray (700, 3)
        Real position [x, y, z] in meters
    initial_guess : ndarray (3,)
        Initial position guess
    guess_name : str
        Name of guess ("A" or "B")
    
    Returns:
    --------
    estimated_states : ndarray (700, 3)
        Estimated states (positions only)
    errors : ndarray (700,)
        Position errors at each timestep
    """
    print ("Experiment C with Constant Position Model")
    print()

    x0 = GUESS_A  # Initial position guess
    P0 = np.diag([1.0, 1.0, 1.0])  # Initial covariance for position only
    F, Q = constant_position_model(DT, q_std=0.075)
    R = np.diag([5.0, 5.0]) # said at the challenge 

    def h_function(x):
            """Proyección de cámara: [x, y, z] → [u, v]"""
            # Expandir x de 3D a 6D para camera_projection
            x_expanded = np.concatenate([x, np.zeros(3)])
            return camera_projection(x_expanded, FOCAL_LENGTH, CX, CY)
        
    def jacobian_function(x):
            """Jacobiano: retorna matriz 2×3"""
            # Expandir a 6D
            x_expanded = np.concatenate([x, np.zeros(3)])
            # Calcular jacobiano completo (2×6)
            H_full = camera_jacobian(x_expanded, FOCAL_LENGTH)
            # Tomar solo columnas de posición (2×3)
            H_reduced = H_full[:, 0:3]
            return H_reduced

    ekf = ExtendedKalmanFilter(x0, P0, F, Q, R, h_function, jacobian_function)
    error_init = np.linalg.norm(GUESS_A - ground_truth[0])
    print(f"Error initial guess: {error_init:.4f} m ({error_init*100:.1f} cm)")
    print()

    n_mesurments = len(measurments)
    print ("Starting Experiment C: Camera 2D + Guess A with Constant Position Model")

    for i in range(n_mesurments):
        z = measurments[i]
        ekf.predict()
        ekf.update(z)

        if (i + 1) % 100 == 0 or i == n_mesurments - 1:
            print(f"  Time Step {i + 1}/{n_mesurments} completed.")
    print ("Filter completed.")
    print ()

    estimated_states = ekf.get_state_history()
    errors = []
    for i in range(len(estimated_states)):
        estimated_pos = estimated_states[i]  # Ya es 3D, no necesita slicing
        true_pos = ground_truth[i]
        error = np.linalg.norm(estimated_pos - true_pos)
        errors.append(error)
    errors = np.array(errors)

    print("="*70)
    print("Results - EXPERIMENT C: Cámara + Posición Constante + Guess A")
    print("="*70)
    print(f"Error initial guess:     {error_init:.4f} m  ({error_init*100:.1f} cm)")
    print(f"Error after 1st meas:    {errors[0]:.4f} m  ({errors[0]*100:.1f} cm)")
    print(f"Mean Error:              {errors.mean():.4f} m  ({errors.mean()*100:.1f} cm)")
    print(f"Standard Deviation:      {errors.std():.4f} m")
    print(f"Error max:               {errors.max():.4f} m")
    print(f"Error min:               {errors.min():.4f} m")
    print(f"Error final:             {errors[-1]:.4f} m  ({errors[-1]*100:.1f} cm)")
    print()
    
    return estimated_states, errors


def experiment_a(measurments, ground_truth): 
    """
    Experiment 1: 3D Sensor + Guess A

    Parameters: 
    messurments: ndarray (700, 3)
        3D sensor measurements
    ground_truth: ndarray (700, 3)

    
    Returns:
    estimated_states: ndarray (700, 6)
        Estimated states over time
    errors: ndarray (700, )
        Position estimation errors over time
    """

    X0 = np.concatenate([GUESS_A, np.zeros(3)])  # Initial state with zero velocity
    P0 = np.eye(6) * 1.0  # Initial covariance

    F, Q = create_motion_model(DT, Q_STD)

    #Sensor model H for 3D position sensor

    H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ])
    #This is the direct MEasurment of the postion 

    R = np.diag([0.02, 0.02, 0.02])  # Measurement noise covariance given by the challenge
    print(f"  Ruido del sensor: σ = {np.sqrt(np.diag(R)[0]):.4f} m (≈14 cm)")
    print()

    # Creating Kalman Filter instance

    kf = KalmanFilter(X0, P0, F, Q, H, R)
    print("Creating Kalman Filter")
    print()

    error_init = np.linalg.norm(GUESS_A - ground_truth[0])
    print(f"  Error del initial guess: {error_init:.4f} m ({error_init*100:.1f} cm)")

    print ("Starting Experiment 1: 3D Sensor + Guess A")
    n_mesurments = len(measurments)

    for i in range(n_mesurments):
        z = measurments[i]
        kf.predict()
        kf.update(z)

        # Showing progress
        if (i + 1) % 100 == 0 or i == n_mesurments - 1:
            print(f"  Time Step {i + 1}/{n_mesurments} completed.")

    print ("Filter completed.")
    print () 
            # Todos los estados estimados
    estimated_states = kf.get_state_history()
    
    # Calcular errores
    errors = []
    for i in range(len(estimated_states)):
        estimated_pos = estimated_states[i, 0:3]
        true_pos = ground_truth[i]
        error = np.linalg.norm(estimated_pos - true_pos)
        errors.append(error)
    
    errors = np.array(errors)
    
 
    print("="*70)
    print("Results - EXPERIMENT 3")
    print("="*70)
    print(f"Error initial guess: {error_init:.4f} m  ({error_init*100:.1f} cm)")
    print(f"Error After first measurement:   {errors[0]:.4f} m  ({errors[0]*100:.1f} cm)")
    print(f"Mean Error:          {errors.mean():.4f} m  ({errors.mean()*100:.1f} cm)")
    print(f"Standard Deviation:     {errors.std():.4f} m")
    print(f"Error max:            {errors.max():.4f} m")
    print(f"Error min:            {errors.min():.4f} m")
    print(f"Error final:             {errors[-1]:.4f} m  ({errors[-1]*100:.1f} cm)")
    print()
    
    return estimated_states, errors

def experiment_b(measurments, ground_truth): 

    """
    Experiment 1: 3D Sensor + Guess A

    Parameters: 
    messurments: ndarray (700, 3)
        3D sensor measurements
    ground_truth: ndarray (700, 3)

    
    Returns:
    estimated_states: ndarray (700, 6)
        Estimated states over time
    errors: ndarray (700, )
        Position estimation errors over time
    """

    X0 = np.concatenate([GUESS_B, np.zeros(3)])  # Initial state with zero velocity
    P0 = np.eye(6) * 1.0  # Initial covariance

    F, Q = create_motion_model(DT, Q_STD)

    #Sensor model H for 3D position sensor

    H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ])
    #This is the direct MEasurment of the postion 

    R = np.diag([0.02, 0.02, 0.02])  # Measurement noise covariance given by the challenge
    print(f"  Sensor noise: σ = {np.sqrt(np.diag(R)[0]):.4f} m (≈14 cm)")
    print()

    # Creating Kalman Filter instance

    kf = KalmanFilter(X0, P0, F, Q, H, R)
    print("Creating Kalman Filter")
    print()

    #Intial guess
    error_init = np.linalg.norm(GUESS_B - ground_truth[0])
    print(f"  Error del initial guess: {error_init:.4f} m ({error_init*100:.1f} cm)")
    print()
    print ("Starting Experiment 2: 3D Sensor + Guess B")
    n_mesurments = len(measurments)

    for i in range(n_mesurments):
        z = measurments[i]
        kf.predict()
        kf.update(z)

        # Showing progress
        if (i + 1) % 100 == 0 or i == n_mesurments - 1:
            print(f"  Time Step {i + 1}/{n_mesurments} completed.")

    print ("Filter completed.")
    print () 
            # Todos los estados estimados
    estimated_states = kf.get_state_history()
    
    # Calcular errores
    errors = []
    for i in range(len(estimated_states)):
        estimated_pos = estimated_states[i, 0:3]
        true_pos = ground_truth[i]
        error = np.linalg.norm(estimated_pos - true_pos)
        errors.append(error)
    
    errors = np.array(errors)
    
   
    print("="*70)
    print("Results - EXPERIMENT 2")
    print("="*70)
    print(f"Error initial guess: {error_init:.4f} m  ({error_init*100:.1f} cm)")
    print(f"Error After first measurement:   {errors[0]:.4f} m  ({errors[0]*100:.1f} cm)")
    print(f"Mean Error:          {errors.mean():.4f} m  ({errors.mean()*100:.1f} cm)")
    print(f"Standard Deviation:     {errors.std():.4f} m")
    print(f"Error max:            {errors.max():.4f} m")
    print(f"Error min:            {errors.min():.4f} m")
    print(f"Error final:             {errors[-1]:.4f} m  ({errors[-1]*100:.1f} cm)")
    print()
    
    return estimated_states, errors

def experiment_c(measurements, ground_truth): 
    """
    Experimento 3: Extended Kalman Filter con Cámara 2D + Guess A
    
    Parameters:
    -----------
    measurements : ndarray (700, 2)
        Camera measurements [u, v] in pixels
    ground_truth : ndarray (700, 3)
        Real position [x, y, z] in meters

    Returns:
    --------
    estimated_states : ndarray (700, 6)
        Estimated states [x, y, z, vx, vy, vz]
    errors : ndarray (700,)
        Position errors at each timestep
    """
    
    # ========================================================================
    # CONFIGURACIÓN
    # ========================================================================
    
    X0 = np.concatenate([GUESS_A, np.zeros(3)])
    #P0 = np.eye(6) * 0.0001
    P0 = np.diag([0.01, 0.01, 10.0, 0.1, 0.1, 0.1])
    
    F, Q = create_motion_model(DT, q_std=0.01)
    R = np.diag([5.0, 5.0])
    
    # Funciones no lineales
    def h_function(x):
        return camera_projection(x, FOCAL_LENGTH, CX, CY)   
    
    def jacobian_function(x):
        return camera_jacobian(x, FOCAL_LENGTH)
    
    # Crear EKF
    ekf = ExtendedKalmanFilter(X0, P0, F, Q, R, h_function, jacobian_function)
    
    print("="*70)
    print("EXPERIMENT 3: Cámara 2D + Guess A")
    print("="*70)
    print()
    print("Extended Kalman Filter created")
    print()
    
    # Error inicial
    error_init = np.linalg.norm(GUESS_A - ground_truth[0])
    print(f"Error initial guess: {error_init:.4f} m ({error_init*100:.1f} cm)")
    print()
    
    print("Executing Extended Kalman Filter")
    n_measurements = len(measurements)
    
    for i in range(n_measurements):  # ← Loop empieza aquí
        z = measurements[i]
        ekf.predict()
        ekf.update(z)
        
        
        if (i + 1) % 100 == 0 or i == n_measurements - 1:
            print(f"  Time Step {i + 1}/{n_measurements} completed.")
    
    
    print("Filter completed")
    print()
    
    estimated_states = ekf.get_state_history()
    
    errors = []
    for i in range(len(estimated_states)):
        estimated_pos = estimated_states[i, 0:3]
        true_pos = ground_truth[i]
        error = np.linalg.norm(estimated_pos - true_pos)
        errors.append(error)
    
    errors = np.array(errors)
    

    
    print("="*70)
    print("Results - EXPERIMENT 3")
    print("="*70)
    print(f"Error initial guess: {error_init:.4f} m  ({error_init*100:.1f} cm)")
    print(f"Error After first measurement:   {errors[0]:.4f} m  ({errors[0]*100:.1f} cm)")
    print(f"Mean Error:          {errors.mean():.4f} m  ({errors.mean()*100:.1f} cm)")
    print(f"Standard Deviation:     {errors.std():.4f} m")
    print(f"Error max:            {errors.max():.4f} m")
    print(f"Error min:            {errors.min():.4f} m")
    print(f"Error final:             {errors[-1]:.4f} m  ({errors[-1]*100:.1f} cm)")
    print()
    
    return estimated_states, errors

def experiment_d(measurements, ground_truth): 
    """
    Experimento d: Extended Kalman Filter con Cámara 2D + Guess B
    
    Parameters:
    -----------
    measurements : ndarray (700, 2)
        Camera measurements [u, v] in pixels
    ground_truth : ndarray (700, 3)
        Real position [x, y, z] in meters

    Returns:
    --------
    estimated_states : ndarray (700, 6)
        Estimated states [x, y, z, vx, vy, vz]
    errors : ndarray (700,)
        Position errors
    """
    
    # ========================================================================
    # CONFIGURACIÓN
    # ========================================================================
    
    X0 = np.concatenate([GUESS_B, np.zeros(3)])
    #P0 = np.eye(6) * 0.0001
    P0 = np.diag([0.01, 0.01, 10.0, 0.1, 0.1, 0.1])
    
    F, Q = create_motion_model(DT, q_std=0.01)
    R = np.diag([5.0, 5.0])
    
    # Funciones no lineales
    def h_function(x):
        return camera_projection(x, FOCAL_LENGTH, CX, CY)   
    
    def jacobian_function(x):
        return camera_jacobian(x, FOCAL_LENGTH)
    
    # Crear EKF
    ekf = ExtendedKalmanFilter(X0, P0, F, Q, R, h_function, jacobian_function)
    
    print("="*70)
    print("EXPERIMENT 4: Cámara 2D + Guess B")
    print("="*70)
    print()
    print("Extended Kalman Filter created")
    print()
    
    # Error inicial
    error_init = np.linalg.norm(GUESS_B - ground_truth[0])
    print(f"Error initial guess: {error_init:.4f} m ({error_init*100:.1f} cm)")
    print()
    
    # ========================================================================
    # EJECUTAR FILTRO
    # ========================================================================
    
    print("Ejecutando Extended Kalman Filter...")
    n_measurements = len(measurements)
    
    for i in range(n_measurements):  # ← Loop empieza aquí
        z = measurements[i]
        ekf.predict()
        ekf.update(z)
        
        # Mostrar progreso
        if (i + 1) % 100 == 0 or i == n_measurements - 1:
            print(f"  Time Step {i + 1}/{n_measurements} completed.")
    # ← Loop termina aquí
    
    print("Filter completed")
    print()
    

    
    estimated_states = ekf.get_state_history()
    
    errors = []
    for i in range(len(estimated_states)):
        estimated_pos = estimated_states[i, 0:3]
        true_pos = ground_truth[i]
        error = np.linalg.norm(estimated_pos - true_pos)
        errors.append(error)
    
    errors = np.array(errors)
    
    
    print("="*70)
    print("Results - EXPERIMENT 4")
    print("="*70)
    print(f"Error initial guess: {error_init:.4f} m  ({error_init*100:.1f} cm)")
    print(f"Error after first measurement:   {errors[0]:.4f} m  ({errors[0]*100:.1f} cm)")
    print(f"Mean Error:          {errors.mean():.4f} m  ({errors.mean()*100:.1f} cm)")
    print(f"Standard Deviation:     {errors.std():.4f} m")
    print(f"Error max:            {errors.max():.4f} m")
    print(f"Error min:            {errors.min():.4f} m")
    print(f"Error final:             {errors[-1]:.4f} m  ({errors[-1]*100:.1f} cm)")
    print()
    
    return estimated_states, errors

def save_results_and_plot(results_dir_name, experiment_name, states_exp, errors_exp, 
                          ground_truth, initial_guess, sensor_label="Sensor"):
    """
    Save results and create plots for any Kalman Filter experiment.
    
    Parameters:
    -----------
    results_dir_name : str
        Name of results directory (e.g., "results_dir")
    experiment_name : str
        Name of experiment (e.g., "exp1_3d_sensor_guess_a")
    states_exp : ndarray (n, 6)
        Estimated states [x, y, z, vx, vy, vz]
    errors_exp : ndarray (n,)
        Position errors at each timestep
    ground_truth : ndarray (n, 3)
        True positions
    initial_guess : ndarray (3,)
        Initial guess used (e.g., GUESS_A or GUESS_B)
    sensor_label : str, optional
        Label for sensor in plot (default: "Sensor")
    """
    
    # ========================================================================
    # GUARDAR RESULTADOS
    # ========================================================================
    
    results_dir = Path("D:/Dev_Space/estimation_challenge/" + results_dir_name + "/")
    results_dir.mkdir(exist_ok=True)
    
    np.save(results_dir / f'{experiment_name}_states.npy', states_exp)
    np.save(results_dir / f'{experiment_name}_errors.npy', errors_exp)
    
    print(f"Results saved in: {results_dir}")
    print()
    
    # ========================================================================
    # CREAR VISUALIZACIÓN
    # ========================================================================
    
    print("Creating visualization...")
    
    fig = plt.figure(figsize=(15, 5))
    
    # ------------------------------------------------------------------------
    # Plot 1: Trayectoria 3D
    # ------------------------------------------------------------------------
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Ground truth
    ax1.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], 
             'b-', label='Ground Truth', linewidth=2)
    
    # Estimación
    ax1.plot(states_exp[:, 0], states_exp[:, 1], states_exp[:, 2], 
             'r-', label='Estimated (KF)', linewidth=1.5, alpha=0.8)
    
    # Sensor en origen
    ax1.scatter([0], [0], [0], c='yellow', s=300, marker='*', 
                edgecolors='black', linewidths=2, label=sensor_label)
    
    # Initial guess
    ax1.scatter([initial_guess[0]], [initial_guess[1]], [initial_guess[2]], 
                c='orange', s=200, marker='^', edgecolors='black', 
                linewidths=2, label=f'Initial Guess')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.legend()
    ax1.set_title(f'{experiment_name}: 3D Trajectory')
    ax1.grid(True)
    
    # ------------------------------------------------------------------------
    # Plot 2: Error vs tiempo
    # ------------------------------------------------------------------------
    ax2 = fig.add_subplot(132)
    
    ax2.plot(errors_exp, 'g-', linewidth=2, label='Error')
    ax2.axhline(y=errors_exp.mean(), color='r', linestyle='--', linewidth=2,
                label=f'Mean: {errors_exp.mean():.3f}m')
    ax2.fill_between(range(len(errors_exp)),
                      errors_exp.mean() - errors_exp.std(),
                      errors_exp.mean() + errors_exp.std(),
                      alpha=0.2, color='red', label='±1 std')
    
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Error (m)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Error vs Time')
    
    # ------------------------------------------------------------------------
    # Plot 3: Vista XY (Top View)
    # ------------------------------------------------------------------------
    ax3 = fig.add_subplot(133)
    
    # Ground truth
    ax3.plot(ground_truth[:, 0], ground_truth[:, 1], 'b-', 
             label='Ground Truth', linewidth=2)
    
    # Estimación
    ax3.plot(states_exp[:, 0], states_exp[:, 1], 'r-', 
             label='Estimated (KF)', linewidth=1.5, alpha=0.8)
    
    # Sensor en origen
    ax3.scatter([0], [0], c='yellow', s=300, marker='*', 
                edgecolors='black', linewidths=2, label=sensor_label)
    
    # Initial guess
    ax3.scatter([initial_guess[0]], [initial_guess[1]], 
                c='orange', s=200, marker='^',
                edgecolors='black', linewidths=2, label='Initial Guess')
    
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')
    ax3.set_title('XY Top View')
    
    # ------------------------------------------------------------------------
    # Guardar figura
    # ------------------------------------------------------------------------
    plt.tight_layout()
    
    plot_filename = f'{experiment_name}.png'
    plt.savefig(results_dir / plot_filename, dpi=150, bbox_inches='tight')
    
    print(f"✓ Plot saved: {plot_filename}")
    plt.show()
    
    print()
    print("="*70)
    print(f"{experiment_name.upper()} COMPLETED!")
    print("="*70)
    print()




def main (): 
    # Load data
    ground_truth, sensor_3d, camera_2d = load_data()

    states_exp_1b, errors_exp_1a = experiment_a_consPos(sensor_3d, ground_truth)

# Convertir estados 3D → 6D para compatibilidad con plotting
    states_exp_6d_1b = np.hstack([states_exp_1b, np.zeros((len(states_exp_1b), 3))])

    # Ahora sí, guardar y graficar
    save_results_and_plot(
        results_dir_name="results_consPos_A",
        experiment_name="exp_consPos_guess_a",
        states_exp=states_exp_6d_1b,  # ← Usar versión 6D
        errors_exp=errors_exp_1a,
        ground_truth=ground_truth,
        initial_guess=GUESS_A,
        sensor_label="3D Sensor (Const. Pos.)"
    )

    states_exp2b, errors_expb = experiment_b_consPos(sensor_3d, ground_truth)

# Convertir estados 3D → 6D para compatibilidad con plotting
    states_exp_6_2b = np.hstack([states_exp2b, np.zeros((len(states_exp2b), 3))])

    # Ahora sí, guardar y graficar
    save_results_and_plot(
        results_dir_name="results_consPos_B",
        experiment_name="exp_consPos_guess_b",
        states_exp=states_exp_6_2b,  # ← Usar versión 6D
        errors_exp=errors_expb,
        ground_truth=ground_truth,
        initial_guess=GUESS_B,
        sensor_label="3D Sensor (Const. Pos.)"
    )


    states_exp3b, errors_exp3b = experiment_c_consPos(camera_2d, ground_truth)
    # Convertir estados 3D → 6D para compatibilidad con plotting
    states_exp_6_3b = np.hstack([states_exp3b, np.zeros((len(states_exp3b), 3))])
    
    save_results_and_plot(
        results_dir_name="results_3B",
        experiment_name="exp3B_camera_guess_a",
        states_exp=states_exp_6_3b,
        errors_exp=errors_exp3b,
        ground_truth=ground_truth,
        initial_guess=GUESS_A,
        sensor_label="Camera 2D"
    )

    # Run Experiment 1: 3D Sensor + Guess A
    states_exp1, errors_exp1 = experiment_a(sensor_3d, ground_truth)
    
    save_results_and_plot(
    results_dir_name="results_1",
    experiment_name="exp1_3d_sensor_guess_a",
    states_exp=states_exp1,
    errors_exp=errors_exp1,
    ground_truth=ground_truth,
    initial_guess=GUESS_A,
    sensor_label="Sensor 3D")
    
    # Run Experiment 2: 3D Sensor + Guess B
    states_exp2, errors_exp2 = experiment_b(sensor_3d, ground_truth)
    
    save_results_and_plot(
    results_dir_name="results_2",
    experiment_name="exp2_3d_sensor_guess_b",
    states_exp=states_exp2,
    errors_exp=errors_exp2,
    ground_truth=ground_truth,
    initial_guess=GUESS_B,
    sensor_label="Sensor 3D")
    # Run Experiment 3: Camera 2D + Guess A
    states_exp3, errors_exp3 = experiment_c(camera_2d, ground_truth)
    
    save_results_and_plot(
        results_dir_name="results_3",
        experiment_name="exp3_camera_guess_a",
        states_exp=states_exp3,
        errors_exp=errors_exp3,
        ground_truth=ground_truth,
        initial_guess=GUESS_A,
        sensor_label="Camera 2D"
    )

    # Run Experiment 4: Camera 2D + Guess B
    states_exp4, errors_exp4 = experiment_d(camera_2d, ground_truth)
    
    save_results_and_plot(
        results_dir_name="results_4",
        experiment_name="exp4_camera_guess_b",
        states_exp=states_exp4,
        errors_exp=errors_exp4,
        ground_truth=ground_truth,
        initial_guess=GUESS_B,
        sensor_label="Camera 2D"
    )

   


if __name__ == "__main__":
    main ()




