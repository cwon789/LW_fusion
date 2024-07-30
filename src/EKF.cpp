#include "EKF.hpp"
#include <cmath>

EKF::EKF()
{
    state_ << 0, 0, 0;
    P_ = Eigen::Matrix3d::Identity();
    Q_ = Eigen::Matrix3d::Identity() * 0.1;
    R_ = Eigen::Matrix3d::Identity() * 0.1;
    I_ = Eigen::Matrix3d::Identity();
}

void EKF::predict(const Eigen::Vector3d& u, double dt)
{
    double theta = state_(2);
    Eigen::Matrix3d F;
    F << 1, 0, -u(0) * std::sin(theta) * dt,
         0, 1,  u(0) * std::cos(theta) * dt,
         0, 0, 1;

    Eigen::Vector3d B;
    B << std::cos(theta) * dt,
         std::sin(theta) * dt,
         dt;

    state_ += B.cwiseProduct(u); // Element-wise multiplication
    P_ = F * P_ * F.transpose() + Q_;
}

void EKF::update(const Eigen::Vector3d& z)
{
    Eigen::Matrix3d H = Eigen::Matrix3d::Identity();
    Eigen::Vector3d y = z - H * state_;
    Eigen::Matrix3d S = H * P_ * H.transpose() + R_;
    Eigen::Matrix3d K = P_ * H.transpose() * S.inverse(); // Correct matrix operation

    state_ += K * y;
    P_ = (I_ - K * H) * P_;
}

Eigen::Vector3d EKF::getState()
{
    return state_;
}
