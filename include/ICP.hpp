#ifndef ICP_HPP
#define ICP_HPP

#include <vector>
#include <Eigen/Dense>

class ICP
{
public:
    ICP();
    Eigen::Matrix3d align(const std::vector<Eigen::Vector2d>& source, const std::vector<Eigen::Vector2d>& target, int max_iterations = 50, double tolerance = 1e-6);

private:
    Eigen::Vector2d computeCentroid(const std::vector<Eigen::Vector2d>& points);
    Eigen::Matrix2d computeCovariance(const std::vector<Eigen::Vector2d>& source, const std::vector<Eigen::Vector2d>& target, const Eigen::Vector2d& src_centroid, const Eigen::Vector2d& tgt_centroid);
    std::vector<Eigen::Vector2d> findClosestPoints(const std::vector<Eigen::Vector2d>& source, const std::vector<Eigen::Vector2d>& target);
    double computeError(const std::vector<Eigen::Vector2d>& source, const std::vector<Eigen::Vector2d>& target);
};

#endif // ICP_HPP
