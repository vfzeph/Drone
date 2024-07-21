import airsim
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DroneController:
    def __init__(self, client: airsim.MultirotorClient):
        self.client = client
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        logger.info("Drone controller initialized")

    def move_by_velocity_z(self, vx, vy, z, duration):
        """
        Move drone by velocity in x and y direction and target z position
        """
        self.client.moveByVelocityZAsync(vx, vy, z, duration).join()

    def get_collision_info(self):
        return self.client.simGetCollisionInfo()

    def get_orientation(self):
        """
        Get the current orientation of the drone as quaternion components
        """
        orientation = self.client.simGetVehiclePose().orientation
        return (orientation.x_val, orientation.y_val, orientation.z_val, orientation.w_val)

    def get_angular_velocity(self):
        """
        Get the current angular velocity of the drone
        """
        kinematics = self.client.getMultirotorState().kinematics_estimated
        angular_velocity = kinematics.angular_velocity
        return np.array([angular_velocity.x_val, angular_velocity.y_val, angular_velocity.z_val])

    def reset(self):
        """
        Reset the drone to its initial state
        """
        logger.info("Resetting drone")
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        logger.info("Drone reset complete")

    def takeoff(self, altitude: float = 3.0):
        """
        Take off to a specified altitude
        """
        logger.info(f"Taking off to altitude: {altitude}m")
        self.client.takeoffAsync(timeout_sec=5).join()
        self.client.moveToZAsync(-altitude, 1).join()
        logger.info("Takeoff complete")

    def land(self):
        """
        Land the drone
        """
        logger.info("Landing drone")
        self.client.landAsync(timeout_sec=5).join()
        self.client.armDisarm(False)
        logger.info("Landing complete")

    def move_to_position(self, x: float, y: float, z: float, velocity: float = 5):
        """
        Move to a position relative to the drone's starting point
        """
        logger.info(f"Moving to position: x={x}, y={y}, z={z}")
        self.client.moveToPositionAsync(x, y, z, velocity).join()
        logger.info("Move complete")

    def rotate(self, angle: float):
        """
        Rotate the drone by a specified angle in degrees
        """
        logger.info(f"Rotating by {angle} degrees")
        yaw = self.get_orientation()[2] + angle
        self.client.rotateToYawAsync(yaw, timeout_sec=5).join()
        logger.info("Rotation complete")

    def get_position(self):
        """
        Get the current position of the drone
        """
        state = self.client.getMultirotorState()
        return state.kinematics_estimated.position

    def get_orientation(self):
        """
        Get the current orientation of the drone (pitch, roll, yaw)
        """
        state = self.client.getMultirotorState()
        orientation = state.kinematics_estimated.orientation
        return airsim.to_eularian_angles(orientation)

    def get_velocity(self):
        """
        Get the current velocity of the drone
        """
        state = self.client.getMultirotorState()
        return state.kinematics_estimated.linear_velocity

    def get_image(self):
        """
        Get the current image from the drone's camera
        """
        responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        return img_rgb

    def execute_trajectory(self, trajectory: list):
        """
        Execute a series of movement commands
        trajectory: list of (x, y, z) tuples
        """
        logger.info("Executing trajectory")
        for point in trajectory:
            self.move_to_position(*point)
            time.sleep(1)  # Short pause between movements
        logger.info("Trajectory execution complete")

    def emergency_stop(self):
        """
        Immediately stop and hover the drone
        """
        logger.warning("Emergency stop initiated")
        self.client.hoverAsync().join()
        logger.warning("Drone stopped and hovering")

    def __del__(self):
        self.client.enableApiControl(False)
        logger.info("Drone controller deactivated")