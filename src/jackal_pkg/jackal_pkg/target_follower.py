#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TwistStamped
from custom_jackal_interfaces.msg import TargetBoxDimensions
from std_msgs.msg import Bool

import numpy as np

import sensor_msgs_py.point_cloud2 as pc2


import math

class TargetFollowerNode(Node): 

    def __init__(self):
        super().__init__("target_follower") 

        # Initial values of angle and distance of the target 
        self.distance = 1000.0
        self.angle = None

        # Parameter Values 
        self.kd = 0.65
        self.k_theta = 0.65
        self.desired_distance = 0.6
        self.desired_angle = 0.0

        # Subscribers
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            "/j100_0710/sensors/lidar3d_0/points",
            self.lidar_callback,
            10
        )        
        self.target_sub = self.create_subscription(
            TargetBoxDimensions,
            "/j100_0710/target_box_dimensions",
            self.target_callback,
            10
        )
        self.comp_sub = self.create_subscription(
            Bool,
            "/j100_0710/completed",
            self.completed_callback,
            10
        )

        # Publishers
        self.cmd_pub = self.create_publisher(
            TwistStamped,
            "/j100_0710/cmd_vel",
            10
        )
        self.point_cloud_pub = self.create_publisher(
            PointCloud2,
            "/j100_0710/target_3d_points",
            10
        )
        self.close_enough_pub = self.create_publisher(
            Bool,
            "/j100_0710/close_enough",
            10
        )

        # Timers 
        # Main control loop timer (100 Hz)
        self.timer = self.create_timer(0.01, self.velocity_controller)
        # Proximity status update timer (1 Hz)
        self.close_enough_timer = self.create_timer(1.0, self.close_enough_update)

        # Camera Info
        self.fx = 580
        self.fy = 580
        self.cx = 360
        self.cy = 270

        # Target bbox info
        self.x_min = None
        self.x_max = None
        self.y_min = None 
        self.y_max = None

        # Logger
        self.get_logger().info("Target Follower node started")

        # Initialized checks
        self.target_bbox_check = False
        self.camera_info_check = False 

        # Motion history for recovery behavior
        self.last_linear = 0.0
        self.last_angle_sign = 1.0

        # Tracking state
        self.track_initialized = False
        
        # Estimated target position in robot frame
        self.track_x = 0.0
        self.track_y = 0.0

        # Estimated target velocity components
        self.track_vx = 0.0
        self.track_vy = 0.0

        # Timestamp of the last valid tracking update
        self.last_track_time = None

        # Counter for consecutive frames where target was lost
        self.lost_frames = 0

        # Fallback flag to bypass camera when target is too close
        self.use_lidar_only = False

        # Operational and task completion flags
        self.close_enough = False
        self.completed = False




    def velocity_controller(self):
        """
        Computes and publishes the desired angular and linear velocity 
        to effectively track and follow the target.
        """
        # Create a fresh message (all float fields default to 0.0 automatically)
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()

        # Case 1: Target successfully grabbed, stop all motion
        if self.completed:
            self.cmd_pub.publish(cmd)
            return
            
        # Case 2: Target lost, rotate in place to search using last known direction
        if self.distance == 1000.0 or self.angle is None: 
            cmd.twist.angular.z = 0.3 * self.last_angle_sign
            self.cmd_pub.publish(cmd)
            return

        # Case 3: Target visible, compute P-Controller tracking laws
        # V = K_d * (d - d_desired)
        linear_x = self.kd * (self.distance - self.desired_distance)
        linear_x = max(min(linear_x, 0.5), -0.2)  # Apply safety saturation limits

        # ω = k_theta * (angle - angle_desired)
        angular_z = self.k_theta * self.angle

        # Populate control commands
        cmd.twist.linear.x = float(linear_x)
        cmd.twist.angular.z = float(angular_z)
        self.cmd_pub.publish(cmd)

        # Update historical states for recovery behavior
        self.last_linear = linear_x
        self.last_angle_sign = 1.0 if self.angle > 0 else -1.0


    def lidar_callback(self, msg: PointCloud2):
        """
        Main LiDAR processing pipeline. Coordinates point cloud filtering, 
        spatial clustering, state estimation, and target metrics computation.
        """
        self.distance = 1000.0

        # Wait until the first bounding box from the camera is received
        if not self.target_bbox_check:
            return

        now = self.get_clock().now()

        # Compute predicted position based on velocity if tracking is active
        if self.track_initialized:
            dt = (
                now.nanoseconds -
                self.last_track_time.nanoseconds
            ) / 1e9

            pred_x = self.track_x + self.track_vx * dt
            pred_y = self.track_y + self.track_vy * dt
        else:
            dt = 0.0
            pred_x = None
            pred_y = None

        # 1. Filter raw LiDAR points (Camera projection / BBox gating)
        object_points = self.filter_lidar_points(msg, pred_x, pred_y)

        # 2. Extract the spatial target cluster around the anchor point
        closest_cluster = self.extract_target_cluster(object_points, pred_x, pred_y)
        if not closest_cluster:
            return

        # 3. Update tracking positions, velocities and handle gating validation
        success = self.update_tracking_state(closest_cluster, pred_x, pred_y, now, dt)
        if not success:
            return
        
        # 4. Compute final range, angle, and update threshold states
        self.compute_target_metrics()

        # 5. Generate and publish the filtered target point cloud for RViz visualization
        self.publish_target_cloud(msg.header, closest_cluster)
        
        # 6. Finalize timestamp update for the next iteration loop
        self.last_track_time = now


    def target_callback(self, msg: TargetBoxDimensions):
        self.x_max = msg.x_max
        self.x_min = msg.x_min
        self.y_max = msg.y_max
        self.y_min = msg.y_min

        self.target_bbox_check = True

    def completed_callback(self, msg: Bool):
        self.completed = msg.data

    def close_enough_update(self):
        msg = bool()
        msg.data = self.close_enough
        self.close_enough_pub.publish(msg)

    def filter_lidar_points(self, msg: PointCloud2, pred_x, pred_y):
        """
        Transforms raw LiDAR points to camera frame, applies projection, 
        and filters them based on Strategy A (BBox) or Strategy B (Proximity).
        """
        object_points = []

        # Rotation and translation matrices (LiDAR to Camera Frame)
        theta = np.deg2rad(75)
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta),  np.cos(theta), 0.0],
            [0.0,            0.0,           1.0]
        ])
        T = np.array([0.423, 0.130, -0.345])

        # Filter raw LiDAR points
        for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = point

            # Apply full rigid transform to camera frame
            p_lidar = np.array([x, y, z])
            p_cam = R @ p_lidar + T

            x_cam, y_cam, z_cam = p_cam

            # Ignore points that are too close or behind the camera
            if y_cam <= 0.3:
                continue

            # Project 3D points onto the 2D image plane
            u = self.fx * (x_cam / y_cam) + self.cx
            v = self.fy * (z_cam / y_cam) + self.cy

            if not self.use_lidar_only:
                # Strategy A: Use bounding box from the camera
                if self.x_min <= u <= self.x_max and self.y_min <= v <= self.y_max:
                    object_points.append((x, y, z))
            else:
                # Strategy B: Use proximity gating to the expected target position
                if pred_x is not None and pred_y is not None:
                    dx = x - pred_x
                    dy = y - pred_y

                    # FIXED: Reduced radius from 0.5m to 0.3m to prevent capturing 
                    # a person passing closely next to the actual target.
                    if math.sqrt(dx*dx + dy*dy) < 0.3:
                        object_points.append((x, y, z))

        # Check if enough valid points were collected
        if len(object_points) < 5:
            self.lost_frames += 1
            if self.lost_frames > 10:
                self.track_initialized = False
                self.use_lidar_only = False
            return []  # Return empty list to indicate validation failure

        return object_points
    
    def extract_target_cluster(self, object_points, pred_x, pred_y):
        """
        Anchors the cluster near the predicted track position (or closest point if not tracking)
        and gathers points within a 25cm distance window.
        """
        if not object_points:
            return []

        if self.track_initialized:
            # Find the single LiDAR point closest to where we expect the target to be
            closest_point_to_target = min(
                object_points,
                key=lambda p: math.sqrt((p[0] - pred_x)**2 + (p[1] - pred_y)**2)
            )
            # Use this specific anchor point's distance relative to the robot base
            target_d = math.sqrt(closest_point_to_target[0]**2 + closest_point_to_target[1]**2)
        else:
            # Fallback for initial execution (first lock): use point closest to the robot
            target_d = min(math.sqrt(p[0]**2 + p[1]**2) for p in object_points)

        # Build the final cluster around our identified 'target_d' base distance
        closest_cluster = []
        for x, y, z in object_points:
            d = math.sqrt(x * x + y * y)
            # Gather points within a 25cm distance window of the target
            if abs(d - target_d) < 0.25:
                closest_cluster.append((d, x, y, z))

        # Ensure cluster contains a valid amount of points
        if len(closest_cluster) < 3:
            return []  # Return empty list to indicate clustering failure
            

        return closest_cluster
    
    def compute_target_metrics(self):
        """
        Computes the final range (distance) and azimuth angle to the target,
        and updates operational state flags based on range thresholds.
        """
        # Calculate final range to target
        self.distance = math.sqrt(
            self.track_x**2 +
            self.track_y**2
        )

        # Handle fallback state toggles based on range thresholds
        if self.distance < 2.0:
            self.use_lidar_only = True
        elif self.distance > 2.5:
            self.use_lidar_only = False
        
        # Check if the robot is close enough to the target
        if self.distance < 1.0:
            self.close_enough = True
        else: 
            self.close_enough = False

        # Calculate azimuth angle to target
        self.angle = math.atan2(
            self.track_y,
            self.track_x
        )

    def publish_target_cloud(self, header, closest_cluster):
        """
        Extracts raw XYZ coordinates from the cluster, generates a PointCloud2 
        message, and publishes it for visualization.
        """
        if not closest_cluster:
            return

        # Extract only the X, Y, Z coordinates (ignoring the distance 'd' stored at index 0)
        cluster_points = [
            (p[1], p[2], p[3])
            for p in closest_cluster
        ]

        # Create the ROS 2 PointCloud2 message
        target_cloud_msg = pc2.create_cloud_xyz32(
            header,
            cluster_points
        )

        # Publish the filtered target cloud for visualization
        self.point_cloud_pub.publish(target_cloud_msg)

def main(args=None):
    rclpy.init(args=args)
    node = TargetFollowerNode() 
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
