#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <Eigen/Dense>
#include "EKF.hpp"
#include "ICP.hpp"

class EKFNode : public rclcpp::Node
{
public:
    EKFNode()
        : Node("ekf_node"), ekf_(), icp_(), last_time_(0)
    {
        laser_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&EKFNode::laserCallback, this, std::placeholders::_1));

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&EKFNode::odomCallback, this, std::placeholders::_1));

        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/ekf_pose", 10);
    }

private:
    void laserCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        // Convert laser scan to point cloud (2D)
        std::vector<Eigen::Vector2d> current_scan;
        for (size_t i = 0; i < msg->ranges.size(); ++i)
        {
            double angle = msg->angle_min + i * msg->angle_increment;
            double range = msg->ranges[i];
            if (range < msg->range_max)
            {
                current_scan.emplace_back(range * cos(angle), range * sin(angle));
            }
        }

        if (target_scan_.empty())
        {
            target_scan_ = current_scan;  // For initial alignment, use the first scan as target
            return;
        }

        // Perform ICP alignment
        Eigen::Matrix3d icp_result = icp_.align(current_scan, target_scan_);

        // Update EKF with ICP result
        Eigen::Vector3d measurement(icp_result(0, 2), icp_result(1, 2), atan2(icp_result(1, 0), icp_result(0, 0)));
        ekf_.update(measurement);

        // Update target_scan_ with current_scan transformed by icp_result for next iteration
        target_scan_ = current_scan;
        for (auto& point : target_scan_)
        {
            Eigen::Vector3d p(point(0), point(1), 1.0);
            p = icp_result * p;
            point(0) = p(0);
            point(1) = p(1);
        }

        // Publish pose
        geometry_msgs::msg::PoseStamped pose_msg;
        Eigen::Vector3d state = ekf_.getState();
        pose_msg.header.stamp = this->get_clock()->now();
        pose_msg.pose.position.x = state(0);
        pose_msg.pose.position.y = state(1);
        pose_msg.pose.orientation.z = sin(state(2) / 2);
        pose_msg.pose.orientation.w = cos(state(2) / 2);
        pose_pub_->publish(pose_msg);
    }

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        if (last_time_ == 0)
        {
            last_time_ = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
            return;
        }

        double current_time = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        double dt = current_time - last_time_;
        last_time_ = current_time;

        Eigen::Vector3d u;
        u(0) = msg->twist.twist.linear.x;
        u(1) = msg->twist.twist.linear.y;
        u(2) = msg->twist.twist.angular.z;

        ekf_.predict(u, dt);
    }

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laser_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    EKF ekf_;
    ICP icp_;
    double last_time_;
    std::vector<Eigen::Vector2d> target_scan_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EKFNode>());
    rclcpp::shutdown();
    return 0;
}
