#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QTimer>
#include <QRadioButton>
#include <QButtonGroup>
#include <QStatusBar>
#include <opencv2/opencv.hpp>
#include "opticalflow.h"

/**
 * @brief Main window for optical flow video viewer
 */
class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

private slots:
    void openFile();
    void togglePlay();
    void nextFrame();
    void prevFrame();
    void faster();
    void slower();
    void jumpToDialog();
    void updateFrame();
    void setFlowMode();

private:
    void setupUI();
    void createControlButtons(QWidget* parent);
    void createFlowModeControls(QWidget* parent);
    void createVideoDisplays(QWidget* parent);
    void createStatusBar();
    
    void showFrame(int idx);
    void releaseVideo();
    void updatePlayButton();
    
    // UI Components
    QLabel* leftDisplay_;
    QLabel* rightDisplay_;
    QLabel* leftTitle_;
    QLabel* rightTitle_;
    QLabel* statusLabel_;
    QPushButton* playButton_;
    QButtonGroup* flowModeGroup_;
    QRadioButton* denseRadio_;
    QRadioButton* sparseRadio_;
    QTimer* playbackTimer_;
    
    // Video handling
    cv::VideoCapture cap_;
    cv::Mat frameBGR_;
    cv::Mat prevGray_;
    cv::Mat currGray_;
    
    // Playback state
    bool playing_;
    int frameIdx_;
    int totalFrames_;
    double fps_;
    double speed_;
    
    // Optical flow
    OpticalFlow opticalFlow_;
    OpticalFlow::FlowMode flowMode_;
    
    // Display settings
    static constexpr int DISPLAY_W = 640;
    static constexpr int DISPLAY_H = 480;
};

#endif // MAINWINDOW_H
