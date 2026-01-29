"""
Docstring for kalman_filter

The modules implement a Kalman Filter for state estimation in dynamic systems.
- Kalman filter classic: For lineal measurments models 
- Extended Kalman filter: For non-lineal measurments models



"""

import numpy as np
from numpy.linalg import inv

# Helper functions

def create_motion_model(dt, q_std = 0.5):
    """
    Creates state transition matrix F and process noise Q for constant velocity model.
    
    Parameters:
    -----------
    dt : float
        Time step between measurements
    
    Returns:
    --------
    F : ndarray (6, 6)
        State transition matrix
    Q : ndarray (6, 6)
        Process noise covariance matrix
    """
    F = np.array([[1, 0, 0, dt, 0, 0],
                  [0, 1, 0, 0, dt, 0],
                  [0, 0, 1, 0, 0, dt],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])
    
    q = q_std ** 2
    """ Q = q * np.array([[dt**4/4, 0, 0, dt**3/2, 0, 0],
                      [0, dt**4/4, 0, 0, dt**3/2, 0],
                      [0, 0, dt**4/4, 0, 0, dt**3/2],
                      [dt**3/2, 0, 0, dt**2, 0, 0],
                      [0, dt**3/2, 0, 0, dt**2, 0],
                      [0, 0, dt**3/2, 0, 0, dt**2]])"""

    G = np.array([[dt**2/2, 0, 0],
                  [0, dt**2/2, 0],
                  [0, 0, dt**2/2],
                  [dt, 0, 0],
                  [0, dt, 0],
                  [0, 0, dt]])
    Q = q * (G @ G.T)
    
    return F, Q

def constant_position_model(dt, q_std = 0.063):
    """
    Creates state transition matrix F and process noise Q for constant position model.
    
    Parameters:
    -----------
    dt : float
        Time step between measurements
        """
    F = np.eye(3)
    q = q_std ** 2
    Q = q * np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
    return F, Q


def camera_projection(x, f, cx, cy):
    """
    Pinhole camera projection: 3D point -> 2D pixel coordinates.
    
    Parameters:
    -----------
    x : ndarray (6,)
        State vector [x, y, z, vx, vy, vz]
    f : float
        Focal length in pixels
    cx : float
        Principal point x-coordinate
    cy : float
        Principal point y-coordinate
    
    Returns:
    --------
    z : ndarray (2,)
        Projected measurement [u, v] in pixels
    """
    if abs (x[2]) < 1e-6:
        raise ValueError("Z coordinate is too close to zero for projection.") 
    
    x_pos = x[0]
    y_pos = x[1]
    z_pos = x[2]
    
    u = f * (x_pos / z_pos) + cx
    v = f * (y_pos / z_pos) + cy
    return np.array([u, v])

def camera_jacobian(x, f):
    """
    Computes Jacobian of camera projection function.
    
    Parameters:
    -----------
    x : ndarray (6,)
        State vector [x, y, z, vx, vy, vz]
    f : float
        Focal length in pixels
    
    Returns:
    --------
    H : ndarray (2, 6)
        Jacobian matrix
    """

    x_pos = x[0]
    y_pos = x[1]
    z_pos = x[2]

    H = np.array([[f / z_pos, 0.0, -f * x_pos / (z_pos ** 2), 0.0, 0.0, 0.0],
              [0.0, f / z_pos, -f * y_pos / (z_pos ** 2), 0.0, 0.0, 0.0]])
    
    return H



# KALMAN FILTER CLASS (Linear)


#Based on geeks for geeks kalman filter implementation
class KalmanFilter:
    """
    Classical Kalman Filter for linear systems.
    
    Used for the 3D position sensor which has a linear measurement model:
    z = H @ x + v
    
    State vector: x = [x, y, z, vx, vy, vz]^T
    """
    
    def __init__(self, x0, P0, F, Q, H, R):
        """
        Initialize Kalman Filter.
        
        Parameters:
        -----------
        x0 : ndarray (6,)
            Initial state estimate
        P0 : ndarray (6, 6)
            Initial state covariance
        F : ndarray (6, 6)
            State transition matrix
        Q : ndarray (6, 6)
            Process noise covariance
        H : ndarray (measurement_dim, 6)
            Measurement matrix (3x6 for 3D sensor)
        R : ndarray (measurement_dim, measurement_dim)
            Measurement noise covariance (3x3 for 3D sensor)
        """
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R

        self.x = x0.copy()
        self.P = P0.copy()

        #updated predicted state and covariance
        self.x_pred = None
        self.P_pred = None

        # State history for analysis
        self.state_history = []
        self.covariance_history = []

        #store dimnensions
        self.state_dim = x0.shape[0] #should be 6
        # Measurement dimension (handle None for EKF)
        if H is not None:
            self.measurement_dim = H.shape[0]  # 3 for 3D sensor, 2 for camera
        else:
            self.measurement_dim = None  # Will be overridden by EKF

        #Identity matrix for not creating a new one each time
        self.I = np.eye(self.state_dim)

    
    
    def predict(self):
        """
        Prediction step of Kalman Filter.

        Equation: 
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        
        Updates:
        --------
        self.x_pred : Predicted state
        self.P_pred : Predicted covariance
        """
        self.x_pred = self.F @ self.x
        self.P_pred = self.F @ self.P @ self.F.T + self.Q
    
    
    def update(self, z):
        """
        Update step of Kalman Filter (linear measurement model).
         Equations:
            y = z - H @ x_pred               (innovation)
            S = H @ P_pred @ H^T + R         (innovation covariance)
            K = P_pred @ H^T @ inv(S)        (Kalman gain)
            x = x_pred + K @ y               (updated state)
            P = (I - K @ H) @ P_pred         (updated covariance)
        
        Parameters:
        -----------
        z : ndarray (measurement_dim,)
            Measurement vector
        
        Updates:
        --------
        self.x : Updated state estimate
        self.P : Updated covariance
        """
        y = z - self.H @ self.x_pred
        S = self.H @ self.P_pred @ self.H.T + self.R
        K = self.P_pred @ self.H.T @ np.linalg.inv(S)
        #Updating state 
        self.x = self.x_pred + K @ y
        self.P = (self.I - K @ self.H) @ self.P_pred
        self.state_history.append(self.x.copy())
        self.covariance_history.append(self.P.copy())


    def get_state(self):
        """
        Returns current state estimate.
        
        Returns:
        --------
        x : ndarray (6,)
            Current state
        """
        return self.x.copy()
    
    
    def get_covariance(self):
        """
        Returns current state covariance.
        
        Returns:
        --------
        P : ndarray (6, 6)
            Current covariance
        """
        return self.P.copy()

    def get_state_history(self):
        """
        Returns history of state estimates.
        
        Returns:
        --------
        state_history : list of ndarray
            List of state estimates over time
        """
        return np.array(self.state_history)


# ============================================================================
# EXTENDED KALMAN FILTER CLASS (Nonlinear)
# ============================================================================

class ExtendedKalmanFilter(KalmanFilter):
    """
    Extended Kalman Filter for nonlinear systems.
    
    Used for the 2D camera sensor which has a nonlinear measurement model:
    z = h(x) + v
    
    where h(x) is the pinhole projection function.

    we make inheritance from KalmanFilter class and override the update method.
    """
    
    def __init__(self, x0, P0, F, Q, R, h_function, jacobian_function):
        """
        Initialize Extended Kalman Filter.
        
        Parameters:
        -----------
        x0 : ndarray (6,)
            Initial state estimate
        P0 : ndarray (6, 6)
            Initial state covariance
        F : ndarray (6, 6)
            State transition matrix
        Q : ndarray (6, 6)
            Process noise covariance
        R : ndarray (2, 2)
            Measurement noise covariance (2x2 for camera)
        h_function : callable
            Nonlinear measurement function h(x) -> z
        jacobian_function : callable
            Function to compute Jacobian H(x)
        """
        super().__init__(x0, P0, F, Q, H = None, R = R)
        #Super make reference to parent class constructor
        self.h_function = h_function
        self.jacobian_function = jacobian_function

        self.measurement_dim = 2  # Camera provides 2D measurements

    def update(self, z):
        """
        Update step of EKF (nonlinear measurement model).
        
        Overrides parent class method to handle nonlinearity.
        
        Equations:
            H = jacobian_function(x_pred)    (compute Jacobian at predicted state)
            z_pred = h_function(x_pred)      (predicted measurement)
            y = z - z_pred                   (innovation)
            S = H @ P_pred @ H^T + R         (innovation covariance)
            K = P_pred @ H^T @ inv(S)        (Kalman gain)
            x = x_pred + K @ y               (updated state)
            P = (I - K @ H) @ P_pred         (updated covariance)
        
        Parameters:
        -----------
        z : ndarray (2,)
            Measurement vector [u, v] in pixels
        
        Updates:
        --------
        self.x : Updated state estimate
        self.P : Updated covariance
        """
        H = self.jacobian_function(self.x_pred)
        z_pred = self.h_function(self.x_pred)
        y = z - z_pred
        S = H @ self.P_pred @ H.T + self.R
        K = self.P_pred @ H.T @ np.linalg.inv(S)
        #Updating state 
        self.x = self.x_pred + K @ y
        self.P = (self.I - K @ H) @ self.P_pred
        self.state_history.append(self.x.copy())
        self.covariance_history.append(self.P.copy())