#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from yolo_msgs.msg import DetectionArray
from custom_jackal_interfaces.msg import TargetBoxDimensions


class Find3DTargetPointsNode(Node):

    def __init__(self):
        super().__init__("find_3d_target_points")

        # Subscriber for YOLO detections
        self.detection_sub = self.create_subscription(
            DetectionArray,
            "/yolo/detections",
            self.detections_callback,
            10
        )

        # Publisher for target box dimensions
        self.pub = self.create_publisher(
            TargetBoxDimensions,
            "/j100_0710/target_box_dimensions",
            10
        )

        # Target class to filter
        self.target_class = "wheelbarrow"

    def detections_callback(self, detection_msg: DetectionArray):
        best_detection = None
        best_score = -1.0

        # Iterate through detections to find the highest scoring target class
        for detection in detection_msg.detections:
            if detection.class_name != self.target_class:
                continue

            if detection.score > best_score:
                best_score = detection.score
                best_detection = detection

        # If target is not found in the current frame, publish zeros to reset state
        if best_detection is None:
            self.publish_msg(0.0, 0.0, 0.0, 0.0)
            return

        # Extract bounding box geometries
        center_x = best_detection.bbox.center.position.x
        center_y = best_detection.bbox.center.position.y
        bbox_width = best_detection.bbox.size.x
        bbox_height = best_detection.bbox.size.y

        # Calculate min/max coordinates
        x_max = center_x + bbox_width / 2.0
        x_min = center_x - bbox_width / 2.0
        y_max = center_y + bbox_height / 2.0
        y_min = center_y - bbox_height / 2.0

        # Publish dimensions immediately (Event-driven approach)
        self.publish_msg(x_max, x_min, y_max, y_min)

        # Optional debugging log
        # self.get_logger().info(
        #     f"Target: {best_detection.class_name} | Score: {best_score:.2f} | "
        #     f"xmin: {x_min:.1f}, xmax: {x_max:.1f} | ymin: {y_min:.1f}, ymax: {y_max:.1f}"
        # )

    def publish_msg(self, x_max, x_min, y_max, y_min):
        msg = TargetBoxDimensions()

        # Explicitly cast to float to prevent ROS 2 IDL type errors
        msg.x_max = float(x_max)
        msg.x_min = float(x_min)
        msg.y_max = float(y_max)
        msg.y_min = float(y_min)

        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Find3DTargetPointsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()