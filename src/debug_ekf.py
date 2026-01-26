import numpy as np
from kalman_filter import (
    ExtendedKalmanFilter,
    create_motion_model,
    camera_projection,
    camera_jacobian
)

# Cargar primera medición de cámara
camera_2d = np.loadtxt('D:/Dev_Space/estimation_challenge/resources/measurements_2d_camera.csv',
                       delimiter=',', skiprows=1)
ground_truth = np.loadtxt('D:/Dev_Space/estimation_challenge/resources/ground_truth_3d_position.csv',
                          delimiter=',', skiprows=1)

# Configuración
GUESS_A = np.array([-0.7, -0.3, 6.6])
FOCAL_LENGTH = 500
CX = 320
CY = 240
DT = 1.0
Q_STD = 0.5

# Crear modelo
F, Q = create_motion_model(DT, q_std=Q_STD)

# Estado inicial
x0 = np.concatenate([GUESS_A, np.zeros(3)])

# Covarianza inicial
P0 = np.eye(6) * 1.0

# Ruido de medición
R = np.diag([5.0, 5.0])

# Funciones
def h_func(x):
    return camera_projection(x, FOCAL_LENGTH, CX, CY)

def jacobian_func(x):
    return camera_jacobian(x, FOCAL_LENGTH)

# Crear EKF
ekf = ExtendedKalmanFilter(x0, P0, F, Q, R, h_func, jacobian_func)

print("="*70)
print("DEBUG EKF - PRIMERA ITERACIÓN")
print("="*70)
print()

print("CONFIGURACIÓN:")
print(f"  Estado inicial x0: {x0}")
print(f"  P0 diagonal: {np.diag(P0)}")
print(f"  R: {R}")
print(f"  Ground truth[0]: {ground_truth[0]}")
print(f"  Medición real z[0]: {camera_2d[0]}")
print()

# PREDICCIÓN
ekf.predict()
print("DESPUÉS DE PREDICT:")
print(f"  x_pred: {ekf.x_pred}")
print(f"  Posición predicha: {ekf.x_pred[0:3]}")
print()

# Calcular medición esperada
z_pred = h_func(ekf.x_pred)
print("MODELO DE MEDICIÓN:")
print(f"  z_pred = h(x_pred): {z_pred}")
print(f"  z_real: {camera_2d[0]}")
print(f"  Innovación y = z - z_pred: {camera_2d[0] - z_pred}")
print()

# Jacobiano
H = jacobian_func(ekf.x_pred)
print("JACOBIANO:")
print(f"  H shape: {H.shape}")
print(f"  H =")
print(H)
print()

# Covarianza de innovación
S = H @ ekf.P_pred @ H.T + R
print("COVARIANZA DE INNOVACIÓN:")
print(f"  S =")
print(S)
print()

# Ganancia de Kalman
K = ekf.P_pred @ H.T @ np.linalg.inv(S)
print("GANANCIA DE KALMAN:")
print(f"  K shape: {K.shape}")
print(f"  K =")
print(K)
print()

# UPDATE
ekf.update(camera_2d[0])

print("DESPUÉS DE UPDATE:")
print(f"  x: {ekf.x}")
print(f"  Posición estimada: {ekf.x[0:3]}")
print(f"  Error vs ground truth: {np.linalg.norm(ekf.x[0:3] - ground_truth[0]):.4f} m")
print()

# Segunda iteración
print("="*70)
print("SEGUNDA ITERACIÓN")
print("="*70)

ekf.predict()
ekf.update(camera_2d[1])

print(f"  Posición estimada: {ekf.x[0:3]}")
print(f"  Error vs ground truth: {np.linalg.norm(ekf.x[0:3] - ground_truth[1]):.4f} m")
print()

# Tercera iteración
ekf.predict()
ekf.update(camera_2d[2])

print(f"  Posición estimada: {ekf.x[0:3]}")
print(f"  Error vs ground truth: {np.linalg.norm(ekf.x[0:3] - ground_truth[2]):.4f} m")