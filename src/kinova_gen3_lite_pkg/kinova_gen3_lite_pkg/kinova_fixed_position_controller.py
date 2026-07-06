import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class KinovaFixedPositionController(Node):
    def __init__(self):
        super().__init__("kinova_fixed_position_controller")
        
        self.get_logger().info("Node initialized. Waiting for gripper completion signal...")

        # Publishers
        self.arm_pub = self.create_publisher(
            JointTrajectory,
            "/j100_0710/manipulators/arm_0_joint_trajectory_controller/joint_trajectory",
            10
        )

        self.complete_pub = self.create_publisher(
            Bool,
            "/j100_0710/completed",
            10
        )
        
        
        # Subscriber to monitor when the gripper action is done
        self.gripper_sub = self.create_subscription(
            Bool,
            "/j100_0710/gripper_done",
            self.gripper_callback,
            10
        )
        
        # Flag to prevent duplicate trajectory dispatches
        self.has_moved = False

    def gripper_callback(self, msg: Bool):
        if msg.data and not self.has_moved:
            self.get_logger().info("Received 'True' from gripper. Executing fixed trajectory...")
            self.move_to_fixed_pose()
            self.has_moved = True  # Latch the flag so it triggers only once

    def move_to_fixed_pose(self):
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
        
        # Insert the 6 joint angles corresponding to your target fixed position
        point.positions = [0.0, 0.0, 0.0, 1.57, 1.57, -1.57] 
        
        # Duration to reach the target position (e.g., 4.0 seconds)
        duration = 4.0
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        
        msg.points.append(point)
        
        self.arm_pub.publish(msg)

        msg_comp = Bool()
        msg_comp.data = True
        self.complete_pub.publish(msg_comp)
        self.get_logger().info("Fixed trajectory published successfully.")


def main(args=None):
    rclpy.init(args=args)
    node = KinovaFixedPositionController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()