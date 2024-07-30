#include "ICP.hpp"
#include <limits>
#include <stdexcept>
#include <cmath>

ICP::ICP()
{}

Eigen::Matrix3d ICP::align(const std::vector<Eigen::Vector2d>& source, const std::vector<Eigen::Vector2d>& target, int max_iterations, double tolerance)
{
    if (source.empty() || target.empty())
    {
        throw std::runtime_error("Source and target point sets must not be empty.");
    }

    Eigen::Matrix3d transform = Eigen::Matrix3d::Identity();
    std::vector<Eigen::Vector2d> transformed_source = source;

    for (int iter = 0; iter < max_iterations; ++iter)
    {
        // Find closest points
        std::vector<Eigen::Vector2d> closest_points = findClosestPoints(transformed_source, target);

        // Compute centroids
        Eigen::Vector2d src_centroid = computeCentroid(transformed_source);
        Eigen::Vector2d tgt_centroid = computeCentroid(closest_points);

        // Compute covariance
        Eigen::Matrix2d W = computeCovariance(transformed_source, closest_points, src_centroid, tgt_centroid);

        // SVD
        Eigen::JacobiSVD<Eigen::Matrix2d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix2d R = svd.matrixV() * svd.matrixU().transpose();

        if (R.determinant() < 0)
        {
            Eigen::Matrix2d V = svd.matrixV();
            V.col(1) *= -1;
            R = V * svd.matrixU().transpose();
        }

        Eigen::Vector2d t = tgt_centroid - R * src_centroid;

        Eigen::Matrix3d iteration_transform = Eigen::Matrix3d::Identity();
        iteration_transform.block<2,2>(0,0) = R;
        iteration_transform.block<2,1>(0,2) = t;

        // Apply the transformation
        for (size_t i = 0; i < transformed_source.size(); ++i)
        {
            transformed_source[i] = (R * source[i]) + t;
        }

        transform = iteration_transform * transform;

        // Check for convergence
        double error = computeError(transformed_source, closest_points);
        if (error < tolerance)
        {
            break;
        }
    }

    return transform;
}

Eigen::Vector2d ICP::computeCentroid(const std::vector<Eigen::Vector2d>& points)
{
    Eigen::Vector2d centroid(0, 0);
    for (const auto& point : points)
    {
        centroid += point;
    }
    centroid /= points.size();
    return centroid;
}

Eigen::Matrix2d ICP::computeCovariance(const std::vector<Eigen::Vector2d>& source, const std::vector<Eigen::Vector2d>& target, const Eigen::Vector2d& src_centroid, const Eigen::Vector2d& tgt_centroid)
{
    Eigen::Matrix2d covariance = Eigen::Matrix2d::Zero();
    for (size_t i = 0; i < source.size(); ++i)
    {
        covariance += (source[i] - src_centroid) * (target[i] - tgt_centroid).transpose();
    }
    return covariance;
}

std::vector<Eigen::Vector2d> ICP::findClosestPoints(const std::vector<Eigen::Vector2d>& source, const std::vector<Eigen::Vector2d>& target)
{
    std::vector<Eigen::Vector2d> closest_points;
    closest_points.reserve(source.size());

    for (const auto& src_point : source)
    {
        double min_dist = std::numeric_limits<double>::max();
        Eigen::Vector2d closest_point;

        for (const auto& tgt_point : target)
        {
            double dist = (src_point - tgt_point).squaredNorm();
            if (dist < min_dist)
            {
                min_dist = dist;
                closest_point = tgt_point;
            }
        }

        closest_points.push_back(closest_point);
    }

    return closest_points;
}

double ICP::computeError(const std::vector<Eigen::Vector2d>& source, const std::vector<Eigen::Vector2d>& target)
{
    double error = 0.0;
    for (size_t i = 0; i < source.size(); ++i)
    {
        error += (source[i] - target[i]).squaredNorm();
    }
    return error / source.size();
}
