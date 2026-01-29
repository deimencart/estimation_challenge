import numpy as np 
import matplotlib.pyplot as plt


print ("This is for obtaining the compaiing information of all the experiments")
print ()

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

    #EXPERIMENTS 
    #EXPERIMENT 1A: 3D Sensor + Guess A with Constant Velocity Model
    experiment_1a = np.load(file_path +'results_1/exp1_3d_sensor_guess_a_states.npy')
    #EXPERIMENT 1B: 3D Sensor + Guess A with Constant Position Model
    experiment_1b = np.load(file_path +'results_consPos_A/exp_consPos_guess_a_states.npy')
    
    #EXPERIMENT 2B: 3D Sensor + Guess B with Constant Position Model
    experiment_2b = np.load(file_path +'results_consPos_B/exp_consPos_guess_b_states.npy')

    #EXPERIMENT 2A: 3D Sensor + Guess B with Constant Velocity
    experiment_2a = np.load(file_path +'results_2/exp2_3d_sensor_guess_b_states.npy')

    #EXPERIMENT 3: Camera 2D + Guess A 
    experiment_3 = np.load(file_path +'results_3/exp3_camera_guess_a_states.npy')
    #EXPERIMENT 4: Camera 2D + Guess B 
    experiment_4 = np.load(file_path +'results_4/exp4_camera_guess_b_states.npy')

    
    ground_truth = np.loadtxt(ground_truth_file, delimiter=',', skiprows=1)
    sensor_3d = np.loadtxt(sensor_3d_file, delimiter=',', skiprows=1)
    camera_2d = np.loadtxt(camera_2d_file, delimiter=',', skiprows=1)


    
    print("Ground Truth Shape:", ground_truth.shape)
    print("3D Sensor Measurements Shape:", sensor_3d.shape)
    print("2D Camera Measurements Shape:", camera_2d.shape)
    print()
    data = {
        # Ground truth and measurements
        'ground_truth': ground_truth,
        'sensor_3d': sensor_3d,
        'camera_2d': camera_2d,
        
        # 3D Sensor experiments
        'exp1a': experiment_1a,  # 3D + Guess A + Velocity
        'exp1b': experiment_1b,  # 3D + Guess A + Position
        'exp2a': experiment_2a,  # 3D + Guess B + Velocity
        'exp2b': experiment_2b,  # 3D + Guess B + Position
        
        # Camera 2D experiments
        'exp3': experiment_3,    # Camera + Guess A
        'exp4': experiment_4,    # Camera + Guess B
    }

    # Print shapes for verification
    print("GROUND TRUTH & MEASUREMENTS:")
    print(f"  Ground Truth:        {ground_truth.shape}")
    print(f"  3D Sensor:           {sensor_3d.shape}")
    print(f"  2D Camera:           {camera_2d.shape}")
    print()
    print("EXPERIMENTS (3D SENSOR):")
    print(f"  Exp 1A (A + Vel):    {experiment_1a.shape}")
    print(f"  Exp 1B (A + Pos):    {experiment_1b.shape}")
    print(f"  Exp 2A (B + Vel):    {experiment_2a.shape}")
    print(f"  Exp 2B (B + Pos):    {experiment_2b.shape}")
    print()
    print("EXPERIMENTS (CAMERA 2D):")
    print(f"  Exp 3 (Cam + A):     {experiment_3.shape}")
    print(f"  Exp 4 (Cam + B):     {experiment_4.shape}")
    print()
    
    return data 


def calculate_errors(data): 
    """
    dic"""

    pass

def main():
    data = load_data()
    # Further processing can be done here

if __name__ == "__main__":
    main()