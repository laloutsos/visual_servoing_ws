import rclpy
from rclpy.node import Node
import math
import numpy as np

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import PointCloud2

# IMPORT YOUR IK FILE
from kinova_gen3_lite_pkg.Kinova_Gen3_Lite_IKP import kgen3_lite_ik

import sensor_msgs_py.point_cloud2 as pc2
from control_msgs.msg import JointTrajectoryControllerState
from custom_jackal_interfaces.msg import TargetHeight
from std_msgs.msg import Float32

class KinovaIKController(Node):
    def __init__(self):
        super().__init__("kinova_ik_controller")

        # Publishers
        self.arm_pub = self.create_publisher(
            JointTrajectory,
            "/j100_0710/manipulators/arm_0_joint_trajectory_controller/joint_trajectory",
            10
        )

        self.height_pub = self.create_publisher(
            Float32,
            "/j100_0710/target_height",
            10
        )

        # Subscribers
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            "/j100_0710/target_3d_points",
            self.lidar_callback,
            10
        )

        self.joint_sub = self.create_subscription(
            JointTrajectoryControllerState,
            "/j100_0710/platform/dynamic_joint_states",
            self.joint_callback,
            10
        )

        # State Variables
        self.current_joints = None
        self.sent = False
        
        self.get_logger().info("Kinova IK Controller started. Waiting for PointCloud2 data...")

    def extract_target_pose(self, msg):
        """
        Parses the point cloud to find the highest point (max Z) and 
        calculates the 6-DOF target pose (X, Y, Z, Roll, Pitch, Yaw).
        Returns a tuple with the pose values, or None if the cloud is empty.
        """
        highest_point = None
        max_z = -float('inf')

        # Extract points from the cloud to find the highest one
        for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = point
            if z > max_z:
                max_z = z
                highest_point = (x, y, z)

        if highest_point is None:
            self.get_logger().warn("PointCloud was empty or contained only NaNs.")
            return None

        self.get_logger().info(
            f"Highest point found at: X={highest_point[0]:.3f}, "
            f"Y={highest_point[1]:.3f}, Z={highest_point[2]:.3f}"
        )

        # Calculate target pose relative to the highest point found
        # Z_target = highest_point[2] - 0.03 + 0.43
        # X_target = highest_point[0] - 0.36
        Y_target = highest_point[1] - 0.05
        X_target = 0.4
        Z_target = 0.60

        Roll = 0.0
        Pitch = 1.57
        Yaw = 0.0

        self.get_logger().info(
            f"Target Pose calculated: X={X_target:.3f}, Y={Y_target:.3f}, Z={Z_target:.3f}"
        )

        return X_target, Y_target, Z_target, Roll, Pitch, Yaw
    

    def solve_inverse_kinematics(self, x, y, z, roll, pitch, yaw):
        """
        Calls the IK solver for the given 6-DOF pose, evaluates the solutions,
        and selects the optimal joint configuration closest to the current joint states.
        Returns the chosen joint configuration list, or None if the solver fails.
        """
        solutions = kgen3_lite_ik(x, y, z, roll, pitch, yaw)

        if solutions is None or len(solutions) == 0:
            self.get_logger().error("IK solver failed to find any solutions.")
            return None

        if self.current_joints is None:
            self.get_logger().warn("No current joint state available, selecting the first IK solution.")
            q = solutions[0]
        else:
            # Choose the closest joint configuration solution to minimize physical arm movement
            def dist(a, b):
                return sum((ai - bi) ** 2 for ai, bi in zip(a, b))

            q = min(solutions, key=lambda sol: dist(sol, self.current_joints))

        self.get_logger().info(f"Selected optimal IK configuration: {q}")

        # Safety check on the chosen joint configuration size
        if q is None or len(q) != 6:
            self.get_logger().error("IK output validation failed (invalid length or null).")
            return None

        return q

    def lidar_callback(self, msg):
        if self.sent:
            return

        self.get_logger().info("Received PointCloud2! Extracting target pose...")

        # 1. Process point cloud and get target 6-DOF pose coordinates
        target_pose = self.extract_target_pose(msg)
        if target_pose is None:
            return

        X_target, Y_target, Z_target, Roll, Pitch, Yaw = target_pose

        # 2. Compute Inverse Kinematics and select the closest joint configuration
        q = self.solve_inverse_kinematics(X_target, Y_target, Z_target, Roll, Pitch, Yaw)
        if q is None:
            return

        # 3. Send to the controller
        self.send_joint_positions(q, duration_sec=3.0)
        self.sent = True

        # 4. Publish target height status
        height_msg = Float32()
        height_msg.data = Z_target
        self.height_pub.publish(height_msg)

    def send_joint_positions(self, positions, duration_sec=3.0):
        msg = JointTrajectory()
        msg.joint_names = [
            "arm_0_joint_1",
            "arm_0_joint_2",
            "arm_0_joint_3",
            "arm_0_joint_4",
            "arm_0_joint_5",
            "arm_0_joint_6",
        ]

        point = JointTrajectoryPoint()
        point.positions = list(positions)
        point.time_from_start.sec = int(duration_sec)

        msg.points.append(point)
        self.arm_pub.publish(msg)
        self.get_logger().info("Joint trajectory sent successfully!")

    def joint_callback(self, msg):
        # Map joint names to their corresponding position values
        name_to_pos = dict(zip(msg.joint_names, msg.interface_values))

        # Extract only arm joints in the correct kinematic order
        joint_order = [
            "arm_0_joint_1",
            "arm_0_joint_2",
            "arm_0_joint_3",
            "arm_0_joint_4",
            "arm_0_joint_5",
            "arm_0_joint_6",
        ]

        try:
            self.current_joints = [
                name_to_pos[j][0].values[0]  # Arm position value index
                for j in joint_order
            ]
        except Exception as e:
            self.get_logger().warn(f"Joint parsing failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = KinovaIKController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()