import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient  # Added for gripper action client
from rclpy.executors import MultiThreadedExecutor  # Added to prevent deadlocks

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import PointCloud2

from kinova_gen3_lite_pkg.Kinova_Gen3_Lite_IKP import kgen3_lite_ik

import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Bool, Float32

from control_msgs.msg import JointTrajectoryControllerState
from control_msgs.action import GripperCommand  # Added for gripper command action


class KinovaCloseController(Node):
    def __init__(self):
        super().__init__("kinova_close_controller")

        self.get_logger().info("Initializing KinovaCloseController...")

        # Publishers
        self.arm_pub = self.create_publisher(
            JointTrajectory,
            "/j100_0710/manipulators/arm_0_joint_trajectory_controller/joint_trajectory",
            10
        )

        self.gripper_done = self.create_publisher(
            Bool,
            "/j100_0710/gripper_done",
            10
        )

        self.get_logger().info("Arm trajectory publisher created.")

        # Action Client for the Gripper
        self.gripper_client = ActionClient(
            self,
            GripperCommand,
            "/j100_0710/manipulators/arm_0_gripper_controller/gripper_cmd"
        )
        
        # Safe discovery of the action server during node startup
        self.get_logger().info("Waiting for gripper action server...")
        if self.gripper_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().info("Gripper action server connected successfully.")
        else:
            self.get_logger().error("Gripper action server NOT found! Check your robot simulation/hardware.")

        # Subscribers
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            "/j100_0710/target_3d_points",
            self.lidar_callback,
            10
        )

        self.target_height_sub = self.create_subscription(
            Float32,
            "/j100_0710/target_height",
            self.target_height_callback,
            10
        )

        self.close_enough_sub = self.create_subscription(
            Bool,
            "/j100_0710/close_enough",
            self.close_enough_callback,
            10
        )

        self.joint_sub = self.create_subscription(
            JointTrajectoryControllerState,
            "/j100_0710/platform/dynamic_joint_states",
            self.joint_callback,
            10
        )

        # Controller Internal States
        self.current_joints = None
        self.target_height = None

        self.sent = False
        self.is_close_enough = False
        self.gripper_timer = None  # Timer to trigger gripper action safely

        self.get_logger().info(
            "Kinova IK Controller started. Waiting for PointCloud2 data and close enough signal..."
        )

    def close_enough_callback(self, msg: Bool):
        self.is_close_enough = msg.data
        self.get_logger().info(
            f"Received close_enough message -> {self.is_close_enough}"
        )

    def is_pipeline_ready(self):
        """
        Validates if the lidar processing pipeline is ready by checking 
        execution status, proximity conditions, and target height availability.
        Returns True if all checks pass, otherwise False.
        """
        if self.sent:
            return False

        if not self.is_close_enough:
            self.get_logger().info("Robot is not close enough yet. Ignoring PointCloud.")
            return False

        if self.target_height is None:
            self.get_logger().warn("Target height not received yet.")
            return False

        return True

    def extract_highest_point(self, msg):
        """
        Parses the incoming point cloud message to identify and return 
        the coordinates of the single highest spatial point (max Z).
        """
        highest_point = None
        max_z = -float('inf')
        point_count = 0

        for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            point_count += 1
            x, y, z = point

            if z > max_z:
                max_z = z
                highest_point = (x, y, z)

        self.get_logger().info(f"Processed {point_count} valid points from PointCloud.")
        return highest_point

    def solve_target_ik(self, x, y, z, roll, pitch, yaw, reference_joints, target_name):
        """
        Invokes the analytical IK solver for a targeted 6-DOF pose and optimizes
        joint configuration selection based on Euclidean distance proximity to reference states.
        """
        solutions = kgen3_lite_ik(x, y, z, roll, pitch, yaw)

        if solutions is None or len(solutions) == 0:
            self.get_logger().error(f"IK solver returned no solutions for the {target_name} target.")
            return None

        # Distance calculation lambda tracker
        def dist(a, b):
            return sum((ai - bi) ** 2 for ai, bi in zip(a, b))

        if reference_joints is None:
            self.get_logger().warn("No current joint state available. Using first IK solution.")
            q = solutions[0]
        else:
            q = min(solutions, key=lambda sol: dist(sol, reference_joints))
            if target_name == "first":
                self.get_logger().info("Selected first IK solution closest to current configuration.")
        
        return q

    def lidar_callback(self, msg):
        # 1. Pipeline Gating Guard Checks
        if not self.is_pipeline_ready():
            return

        # Mark as sent immediately to prevent multi-threaded duplicate execution
        self.sent = True
        self.get_logger().info(f"Processing PointCloud. Target height = {self.target_height:.3f}")

        # 2. Extract Target Geometry Anchor Point
        highest_point = self.extract_highest_point(msg)
        if highest_point is None:
            self.get_logger().warn("PointCloud was empty or contained only NaNs.")
            self.sent = False
            return

        self.get_logger().info(
            f"Highest point found -> "
            f"X={highest_point[0]:.4f}, "
            f"Y={highest_point[1]:.4f}, "
            f"Z={highest_point[2]:.4f}"
        )

        # Fixed Orientation and Latency States
        Roll, Pitch, Yaw = 0.0, 1.57, 0.0
        Y_target = highest_point[1]

        # 3. Compute and Solve First Sequence Target Pose
        Z_target_1 = 0.60
        X_target_1 = 0.45
        
        self.get_logger().info(
            f"First target pose calculated -> "
            f"X={X_target_1:.4f}, Y={Y_target:.4f}, Z={Z_target_1:.4f}"
        )
        self.get_logger().info("Calling IK solver for the first target...")
        
        q1 = self.solve_target_ik(X_target_1, Y_target, Z_target_1, Roll, Pitch, Yaw, self.current_joints, "first")
        if q1 is None:
            self.sent = False
            return

        # 4. Compute and Solve Second Sequence Target Pose (Advanced offsets)
        X_offset = 0.05
        Z_offset = -0.025  
        X_target_2 = X_target_1 + X_offset
        Z_target_2 = Z_target_1 + Z_offset
        
        self.get_logger().info(
            f"Second target pose calculated -> "
            f"X={X_target_2:.4f} (further forward), "
            f"Y={Y_target:.4f} (stable), "
            f"Z={Z_target_2:.4f} (slightly lower)"
        )
        self.get_logger().info("Calling IK solver for the second target...")
        
        q2 = self.solve_target_ik(X_target_2, Y_target, Z_target_2, Roll, Pitch, Yaw, q1, "second")
        if q2 is None:
            self.sent = False
            return

        # 5. Length Validation Check
        if len(q1) != 6 or len(q2) != 6:
            self.get_logger().error("Invalid IK solutions generated.")
            self.sent = False
            return

        # 6. Execute Multi-point Trajectory Pipeline & Arm Action Timer
        self.get_logger().info("Sending sequenced trajectory to arm controller...")
        self.send_joint_positions(q1, q2, duration_1=3.0, duration_2=4.5)

        # Create a one-shot timer to trigger gripper action exactly at 5.0 seconds
        self.gripper_timer = self.create_timer(4.5, self.timer_gripper_callback)
        self.get_logger().info("Trajectory sent successfully. Gripper closure scheduled at T+4.5s.")

    def send_joint_positions(self, q1, q2, duration_1=3.0, duration_2=5.0):
        msg = JointTrajectory()
        msg.joint_names = [
            "arm_0_joint_1",
            "arm_0_joint_2",
            "arm_0_joint_3",
            "arm_0_joint_4",
            "arm_0_joint_5",
            "arm_0_joint_6",
        ]

        # First waypoint (Reaches target at 3.0 seconds)
        point1 = JointTrajectoryPoint()
        point1.positions = list(q1)
        point1.time_from_start.sec = int(duration_1)
        point1.time_from_start.nanosec = int((duration_1 - int(duration_1)) * 1e9)
        msg.points.append(point1)

        # Second waypoint (Reaches advanced X & lower Z target at 5.0 seconds)
        point2 = JointTrajectoryPoint()
        point2.positions = list(q2)
        point2.time_from_start.sec = int(duration_2)
        point2.time_from_start.nanosec = int((duration_2 - int(duration_2)) * 1e9)
        msg.points.append(point2)

        self.arm_pub.publish(msg)
        self.get_logger().info("Joint trajectory points published successfully.")

    def timer_gripper_callback(self):
        # Cancel the timer so it only executes once
        if self.gripper_timer is not None:
            self.gripper_timer.cancel()
            self.gripper_timer = None
        
        self.get_logger().info("Second movement finished. Triggering gripper closure...")
        self.send_gripper_goal()

    def send_gripper_goal(self):
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = 0.5
        goal_msg.command.max_effort = 10.0

        self.get_logger().info("Sending gripper command goal...")
        self._send_goal_future = self.gripper_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.gripper_response_callback)

    def gripper_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Gripper goal rejected by server.")
            return
        
        self.get_logger().info("Gripper goal accepted by server. Waiting for completion...")
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.gripper_result_callback)

    def gripper_result_callback(self, future):
        msg = Bool()
        msg.data = True

        self.gripper_done.publish(msg)
        self.get_logger().info("Gripper action finished executing completely.")

    def joint_callback(self, msg):
        try:
            name_to_pos = dict(zip(msg.joint_names, msg.interface_values))
            joint_order = [
                "arm_0_joint_1",
                "arm_0_joint_2",
                "arm_0_joint_3",
                "arm_0_joint_4",
                "arm_0_joint_5",
                "arm_0_joint_6",
            ]
            self.current_joints = [name_to_pos[j][0].values[0] for j in joint_order]
        except Exception as e:
            self.get_logger().debug(f"Joint parsing failed: {e}")

    def target_height_callback(self, msg):
        self.target_height = msg.data


def main(args=None):
    rclpy.init(args=args)
    node = KinovaCloseController()

    # Using MultiThreadedExecutor to handle action client feedback concurrently without deadlocks
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    node.get_logger().info("Node spinning with MultiThreadedExecutor...")
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()