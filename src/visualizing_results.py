import numpy as np 

file_path_1 = "D:/Dev_Space/estimation_challenge/results_1/"
file_path_2 = "D:/Dev_Space/estimation_challenge/results_2/"
file_path_3 = "D:/Dev_Space/estimation_challenge/results_3/"

data_3 = np.load(file_path_3 + "exp3_camera_guess_a_states.npy")

print("Datos from first file:") 
print(data_3.shape)
print(data_3[:100])
