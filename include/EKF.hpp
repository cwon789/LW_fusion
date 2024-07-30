#ifndef EKF_HPP
#define EKF_HPP

#include <Eigen/Dense>

class EKF
{
public:
    EKF();
    void predict(const Eigen::Vector3d& u, double dt);
    void update(const Eigen::Vector3d& z);
    Eigen::Vector3d getState();

private:
    Eigen::Vector3d state_; // [x, y, theta]
    Eigen::Matrix3d P_; // Covariance matrix
    Eigen::Matrix3d Q_; // Process noise covariance
    Eigen::Matrix3d R_; // Measurement noise covariance
    Eigen::Matrix3d I_; // Identity matrix
};

#endif // EKF_HPP
