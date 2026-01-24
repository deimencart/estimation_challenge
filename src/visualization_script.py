import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D

file_path = "D:/Dev_Space/estimation_challenge/"
ground_truth_file = file_path + "resources/ground_truth_3d_position.csv"
camera_2d_file = file_path + "resources/measurements_2d_camera.csv"
sensor_3d_file = file_path + "resources/measurements_3d_position_sensor.csv"

ground_truth = np.loadtxt(ground_truth_file, delimiter=',', skiprows=1)
sensor_3d = np.loadtxt(sensor_3d_file, delimiter=',', skiprows=1)
camera_2d = np.loadtxt(camera_2d_file, delimiter=',', skiprows=1)

print ("Ground Truth Shape:", ground_truth.shape)
print ("3D Sensor Measurements Shape:", sensor_3d.shape)
print ("2D Camera Measurements Shape:", camera_2d.shape)


# Define parameters 

# Initial guesses
guess_A = np.array([-0.7, -0.3, 6.6])
guess_B = np.array([-1.8, -0.5, 7.9])

# Camera intrinsic parameters
focal_length = 500  
cx = 320  
cy = 240
image_width = 640
image_height = 480

# Colors for plotting
color_gt = 'blue'
color_sensor3d = 'red'
color_camera = 'green'
color_guess_a = 'orange'
color_guess_b = 'purple'

fig = plt.figure(figsize=(18, 6))
fig.suptitle('Data Visualization - Estimation Problem', 
             fontsize=16)
# ============================================================================
# PLOT 1: 3D VIEW
# ============================================================================

ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.set_title('View 3D: Complete Trayectory', fontsize=12, fontweight='bold')

ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.set_title('View 3D: Complete Trayectory', fontsize=12, fontweight='bold')

print("Creating plot 1: View 3D")

# Ground truth: línea continua azul
ax1.plot(ground_truth[:, 0],  # columna X
         ground_truth[:, 1],  # columna Y
         ground_truth[:, 2],  # columna Z
         color=color_gt,      # 'blue'
         linewidth=2,
         label='Ground Truth',
         alpha=0.8)

# Sensor 3D: puntos rojos (submuestreados cada 10)
step = 10  # mostrar cada 10 puntos
ax1.scatter(sensor_3d[::step, 0],  # cada 10 puntos, columna X
            sensor_3d[::step, 1],  # cada 10 puntos, columna Y
            sensor_3d[::step, 2],  # cada 10 puntos, columna Z
            color=color_sensor3d,  # 'red'
            s=20,                  # tamaño de los puntos
            alpha=0.5,             # transparencia
            label='Sensor 3D (noisy)')

# Punto inicial: marcador verde grande
ax1.scatter(ground_truth[0, 0],    # primer punto, X
            ground_truth[0, 1],    # primer punto, Y
            ground_truth[0, 2],    # primer punto, Z
            color='green',
            s=200,                 # grande
            marker='o',            # círculo
            edgecolors='black',
            linewidth=2,
            label='Inicio',
            zorder=5)              # dibujarlo encima

# Punto final: marcador morado
ax1.scatter(ground_truth[-1, 0],   # último punto [-1], X
            ground_truth[-1, 1],   # último punto, Y
            ground_truth[-1, 2],   # último punto, Z
            color='purple',
            s=200,
            marker='s',            # cuadrado
            edgecolors='black',
            linewidth=2,
            label='Final',
            zorder=5)

# Initial Guess A: triángulo naranja
ax1.scatter(guess_A[0],           # x de guess A
            guess_A[1],           # y de guess A
            guess_A[2],           # z de guess A
            color=color_guess_a,  # 'orange'
            s=150,
            marker='^',           # triángulo hacia arriba
            edgecolors='black',
            linewidth=2,
            label='Guess A',
            zorder=5)

# Initial Guess B: triángulo púrpura
ax1.scatter(guess_B[0],
            guess_B[1],
            guess_B[2],
            color=color_guess_b,  # 'purple'
            s=150,
            marker='^',
            edgecolors='black',
            linewidth=2,
            label='Guess B',
            zorder=5)

# Sensor en origen: estrella amarilla
ax1.scatter(0, 0, 0,
            color='yellow',
            s=300,
            marker='*',           # estrella
            edgecolors='black',
            linewidth=2,
            label='Sensor (0,0,0)',
            zorder=10)

# Etiquetas de los ejes
ax1.set_xlabel('X [m]', fontsize=10)
ax1.set_ylabel('Y [m]', fontsize=10)
ax1.set_zlabel('Z [m]', fontsize=10)

# Leyenda
ax1.legend(fontsize=8, loc='upper left')


# ============================================================================
# PLOT 2: 2D VIEW
# ============================================================================
# Grid
ax1.grid(True, alpha=0.3)
ax2 = fig.add_subplot(1, 3, 2)
ax2.set_title('View XY (Top View): Circular Motion', fontsize=12, fontweight='bold')

print("plot 2: XY View...")

# Ground truth: línea azul
ax2.plot(ground_truth[:, 0],  # X
         ground_truth[:, 1],  # Y
         color=color_gt,
         linewidth=2,
         label='Ground Truth',
         alpha=0.8)

# Sensor 3D: puntos pequeños (submuestreados)
ax2.scatter(sensor_3d[::5, 0],  # cada 5 puntos, X
            sensor_3d[::5, 1],  # cada 5 puntos, Y
            color=color_sensor3d,
            s=10,
            alpha=0.3,
            label='Sensor 3D')

# Punto inicial: círculo verde
ax2.scatter(ground_truth[0, 0],
            ground_truth[0, 1],
            color='green',
            s=150,
            marker='o',
            edgecolors='black',
            linewidth=2,
            zorder=5)

# Guess A
ax2.scatter(guess_A[0], guess_A[1],
            color=color_guess_a,
            s=100,
            marker='^',
            edgecolors='black',
            linewidth=2,
            label='Guess A',
            zorder=5)

# Guess B
ax2.scatter(guess_B[0], guess_B[1],
            color=color_guess_b,
            s=100,
            marker='^',
            edgecolors='black',
            linewidth=2,
            label='Guess B',
            zorder=5)

# Origen
ax2.scatter(0, 0,
            color='yellow',
            s=200,
            marker='*',
            edgecolors='black',
            linewidth=2,
            zorder=10)

ax2.set_xlabel('X [m]', fontsize=10)
ax2.set_ylabel('Y [m]', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.axis('equal')  # Importante: aspecto 1:1 para ver el círculo bien

# ============================================================================
# PLOT 3: MEDICIONES CÁMARA
# ============================================================================

ax3 = fig.add_subplot(1, 3, 3)
ax3.set_title('2d Measurments (Pixel Space)', fontsize=12, fontweight='bold')

print("Creating plot 3: Camera Measurements...")

# Mediciones: puntos con gradiente de color (tiempo)
scatter = ax3.scatter(camera_2d[:, 0],  # U (pixel x)
                      camera_2d[:, 1],  # V (pixel y)
                      c=np.arange(len(camera_2d)),  # color según índice (tiempo)
                      s=5,
                      cmap='viridis',  # mapa de colores
                      alpha=0.6)

# Inicio: verde
ax3.scatter(camera_2d[0, 0], camera_2d[0, 1],
            color='green',
            s=150,
            marker='o',
            edgecolors='black',
            linewidth=2,
            label='Inicio',
            zorder=5)

# Final: púrpura
ax3.scatter(camera_2d[-1, 0], camera_2d[-1, 1],
            color='purple',
            s=150,
            marker='s',
            edgecolors='black',
            linewidth=2,
            label='Final',
            zorder=5)

# Límites de la imagen (640x480)
ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax3.axhline(y=image_height, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax3.axvline(x=image_width, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# Centro de la imagen (cx, cy)
ax3.scatter(cx, cy,
            color='red',
            s=100,
            marker='s',
            linewidth=3,
            label='Centro imagen')

ax3.set_xlabel('U [pixels]', fontsize=10)
ax3.set_ylabel('V [pixels]', fontsize=10)
ax3.set_xlim(-50, image_width + 50)  # margen extra
ax3.set_ylim(image_height + 50, -50)  # IMPORTANTE: Y invertido (origen arriba-izquierda)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# ============================================================================
# PASO 4: AJUSTES FINALES Y GUARDAR
# ============================================================================

plt.tight_layout()
output_file = file_path +'outputs/data_visualization.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print()
print(f"Visualización guardada: {output_file}")

plt.show()