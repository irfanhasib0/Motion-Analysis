#ifndef OPTICALFLOW_H
#define OPTICALFLOW_H

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>

/**
 * @brief Optical flow computation class supporting dense and sparse methods
 */
class OpticalFlow {
public:
    enum FlowMode {
        DENSE,  // Farneback dense optical flow
        SPARSE  // Lucas-Kanade sparse optical flow
    };

    OpticalFlow();
    ~OpticalFlow() = default;

    /**
     * @brief Compute optical flow between two frames
     * @param prevGray Previous grayscale frame
     * @param currGray Current grayscale frame
     * @param currBGR Current BGR frame (used for sparse flow visualization)
     * @param mode Flow computation mode (DENSE or SPARSE)
     * @return RGB visualization of optical flow
     */
    cv::Mat compute(const cv::Mat& prevGray, const cv::Mat& currGray, 
                    const cv::Mat& currBGR, FlowMode mode);

private:
    /**
     * @brief Compute dense optical flow using Farneback method
     * @param prevGray Previous grayscale frame
     * @param currGray Current grayscale frame
     * @return RGB visualization with HSV color coding
     */
    cv::Mat computeDenseFlow(const cv::Mat& prevGray, const cv::Mat& currGray);

    /**
     * @brief Compute sparse optical flow using Lucas-Kanade method
     * @param prevGray Previous grayscale frame
     * @param currGray Current grayscale frame
     * @param currBGR Current BGR frame for background
     * @return RGB visualization with tracked points and trajectories
     */
    cv::Mat computeSparseFlow(const cv::Mat& prevGray, const cv::Mat& currGray,
                              const cv::Mat& currBGR);

    // Farneback dense flow parameters
    double pyrScale_;
    int levels_;
    int winsize_;
    int iterations_;
    int polyN_;
    double polySigma_;

    // Lucas-Kanade sparse flow parameters
    int maxCorners_;
    double qualityLevel_;
    int minDistance_;
    cv::Size lkWinSize_;
    int lkMaxLevel_;
};

#endif // OPTICALFLOW_H
