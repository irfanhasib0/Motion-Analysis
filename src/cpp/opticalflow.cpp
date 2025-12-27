#include "opticalflow.h"
#include <vector>

OpticalFlow::OpticalFlow()
    : pyrScale_(0.5)
    , levels_(3)
    , winsize_(15)
    , iterations_(3)
    , polyN_(5)
    , polySigma_(1.2)
    , maxCorners_(200)
    , qualityLevel_(0.01)
    , minDistance_(7)
    , lkWinSize_(15, 15)
    , lkMaxLevel_(2)
{
}

cv::Mat OpticalFlow::compute(const cv::Mat& prevGray, const cv::Mat& currGray,
                             const cv::Mat& currBGR, FlowMode mode)
{
    if (prevGray.empty() || currGray.empty()) {
        return cv::Mat();
    }

    switch (mode) {
        case DENSE:
            return computeDenseFlow(prevGray, currGray);
        case SPARSE:
            return computeSparseFlow(prevGray, currGray, currBGR);
        default:
            return cv::Mat();
    }
}

cv::Mat OpticalFlow::computeDenseFlow(const cv::Mat& prevGray, const cv::Mat& currGray)
{
    // Compute dense optical flow
    cv::Mat flow;
    cv::calcOpticalFlowFarneback(
        prevGray, currGray, flow,
        pyrScale_, levels_, winsize_,
        iterations_, polyN_, polySigma_, 0
    );

    // Convert flow to polar coordinates
    cv::Mat flowParts[2];
    cv::split(flow, flowParts);
    cv::Mat magnitude, angle;
    cv::cartToPolar(flowParts[0], flowParts[1], magnitude, angle, true);

    // Create HSV visualization
    cv::Mat hsv = cv::Mat::zeros(currGray.size(), CV_8UC3);
    std::vector<cv::Mat> hsvChannels(3);
    
    // Hue from angle (0-360 -> 0-180 for OpenCV)
    angle.convertTo(hsvChannels[0], CV_8U, 0.5);
    
    // Saturation from normalized magnitude
    cv::normalize(magnitude, hsvChannels[1], 0, 255, cv::NORM_MINMAX);
    hsvChannels[1].convertTo(hsvChannels[1], CV_8U);
    
    // Full value
    hsvChannels[2] = cv::Mat::ones(currGray.size(), CV_8U) * 255;
    
    cv::merge(hsvChannels, hsv);

    // Convert HSV to RGB
    cv::Mat rgb;
    cv::cvtColor(hsv, rgb, cv::COLOR_HSV2RGB);
    
    return rgb;
}

cv::Mat OpticalFlow::computeSparseFlow(const cv::Mat& prevGray, const cv::Mat& currGray,
                                       const cv::Mat& currBGR)
{
    // Detect corners to track
    std::vector<cv::Point2f> p0;
    cv::goodFeaturesToTrack(prevGray, p0, maxCorners_, qualityLevel_, 
                            minDistance_, cv::noArray(), 7);

    // Start with grayscale background
    cv::Mat visBGR;
    cv::cvtColor(prevGray, visBGR, cv::COLOR_GRAY2BGR);

    if (p0.empty()) {
        cv::Mat rgb;
        cv::cvtColor(visBGR, rgb, cv::COLOR_BGR2RGB);
        return rgb;
    }

    // Calculate optical flow
    std::vector<cv::Point2f> p1;
    std::vector<uchar> status;
    std::vector<float> err;
    
    cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 0.03);
    cv::calcOpticalFlowPyrLK(prevGray, currGray, p0, p1, status, err,
                             lkWinSize_, lkMaxLevel_, criteria);

    // Draw tracks
    cv::Mat mask = cv::Mat::zeros(currBGR.size(), CV_8UC3);
    
    for (size_t i = 0; i < p0.size(); i++) {
        if (status[i]) {
            cv::Point2i newPt(static_cast<int>(p1[i].x), static_cast<int>(p1[i].y));
            cv::Point2i oldPt(static_cast<int>(p0[i].x), static_cast<int>(p0[i].y));
            
            // Draw line (track)
            cv::line(mask, oldPt, newPt, cv::Scalar(0, 255, 0), 2);
            
            // Draw circle (current point)
            cv::circle(visBGR, newPt, 3, cv::Scalar(0, 0, 255), -1);
        }
    }

    // Combine mask and visualization
    cv::add(visBGR, mask, visBGR);

    // Convert to RGB
    cv::Mat rgb;
    cv::cvtColor(visBGR, rgb, cv::COLOR_BGR2RGB);
    
    return rgb;
}
