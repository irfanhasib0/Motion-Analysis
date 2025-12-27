#include "mainwindow.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QInputDialog>
#include <QMessageBox>
#include <QImage>
#include <QPixmap>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , leftDisplay_(nullptr)
    , rightDisplay_(nullptr)
    , leftTitle_(nullptr)
    , rightTitle_(nullptr)
    , statusLabel_(nullptr)
    , playButton_(nullptr)
    , flowModeGroup_(nullptr)
    , denseRadio_(nullptr)
    , sparseRadio_(nullptr)
    , playbackTimer_(nullptr)
    , playing_(false)
    , frameIdx_(0)
    , totalFrames_(0)
    , fps_(25.0)
    , speed_(1.0)
    , flowMode_(OpticalFlow::DENSE)
{
    setupUI();
    
    // Setup playback timer
    playbackTimer_ = new QTimer(this);
    connect(playbackTimer_, &QTimer::timeout, this, &MainWindow::updateFrame);
}

MainWindow::~MainWindow()
{
    releaseVideo();
}

void MainWindow::setupUI()
{
    setWindowTitle("Video + Optical Flow Viewer (C++ Qt)");
    resize(1350, 800);
    
    // Central widget
    QWidget* centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    
    QVBoxLayout* mainLayout = new QVBoxLayout(centralWidget);
    mainLayout->setSpacing(10);
    mainLayout->setContentsMargins(10, 10, 10, 10);
    
    // Control buttons
    QWidget* controlFrame = new QWidget();
    createControlButtons(controlFrame);
    mainLayout->addWidget(controlFrame);
    
    // Flow mode controls
    QWidget* modeFrame = new QWidget();
    createFlowModeControls(modeFrame);
    mainLayout->addWidget(modeFrame);
    
    // Video displays
    QWidget* displayFrame = new QWidget();
    createVideoDisplays(displayFrame);
    mainLayout->addWidget(displayFrame);
    
    // Status bar
    createStatusBar();
}

void MainWindow::createControlButtons(QWidget* parent)
{
    QHBoxLayout* layout = new QHBoxLayout(parent);
    layout->setSpacing(5);
    
    QPushButton* openBtn = new QPushButton("Open", parent);
    connect(openBtn, &QPushButton::clicked, this, &MainWindow::openFile);
    layout->addWidget(openBtn);
    
    playButton_ = new QPushButton("Play", parent);
    connect(playButton_, &QPushButton::clicked, this, &MainWindow::togglePlay);
    layout->addWidget(playButton_);
    
    QPushButton* prevBtn = new QPushButton("<<", parent);
    connect(prevBtn, &QPushButton::clicked, this, &MainWindow::prevFrame);
    layout->addWidget(prevBtn);
    
    QPushButton* nextBtn = new QPushButton(">>", parent);
    connect(nextBtn, &QPushButton::clicked, this, &MainWindow::nextFrame);
    layout->addWidget(nextBtn);
    
    QPushButton* fasterBtn = new QPushButton("Faster", parent);
    connect(fasterBtn, &QPushButton::clicked, this, &MainWindow::faster);
    layout->addWidget(fasterBtn);
    
    QPushButton* slowerBtn = new QPushButton("Slower", parent);
    connect(slowerBtn, &QPushButton::clicked, this, &MainWindow::slower);
    layout->addWidget(slowerBtn);
    
    QPushButton* jumpBtn = new QPushButton("Jump To", parent);
    connect(jumpBtn, &QPushButton::clicked, this, &MainWindow::jumpToDialog);
    layout->addWidget(jumpBtn);
    
    layout->addStretch();
}

void MainWindow::createFlowModeControls(QWidget* parent)
{
    QHBoxLayout* layout = new QHBoxLayout(parent);
    layout->setSpacing(10);
    
    QLabel* modeLabel = new QLabel("Flow Mode:", parent);
    layout->addWidget(modeLabel);
    
    flowModeGroup_ = new QButtonGroup(parent);
    
    denseRadio_ = new QRadioButton("Dense", parent);
    denseRadio_->setChecked(true);
    flowModeGroup_->addButton(denseRadio_, 0);
    layout->addWidget(denseRadio_);
    
    sparseRadio_ = new QRadioButton("Sparse", parent);
    flowModeGroup_->addButton(sparseRadio_, 1);
    layout->addWidget(sparseRadio_);
    
    connect(flowModeGroup_, SIGNAL(buttonClicked(int)),
            this, SLOT(setFlowMode()));
    
    layout->addStretch();
}

void MainWindow::createVideoDisplays(QWidget* parent)
{
    QVBoxLayout* mainLayout = new QVBoxLayout(parent);
    QHBoxLayout* layout = new QHBoxLayout();
    
    // Left display (original video)
    QVBoxLayout* leftLayout = new QVBoxLayout();
    leftTitle_ = new QLabel("Original Video", parent);
    leftTitle_->setAlignment(Qt::AlignCenter);
    leftLayout->addWidget(leftTitle_);
    
    leftDisplay_ = new QLabel(parent);
    leftDisplay_->setFixedSize(DISPLAY_W, DISPLAY_H);
    leftDisplay_->setStyleSheet("QLabel { background-color: black; }");
    leftDisplay_->setAlignment(Qt::AlignCenter);
    leftLayout->addWidget(leftDisplay_);
    
    layout->addLayout(leftLayout);
    layout->addSpacing(20);
    
    // Right display (optical flow)
    QVBoxLayout* rightLayout = new QVBoxLayout();
    rightTitle_ = new QLabel("Optical Flow - Dense", parent);
    rightTitle_->setAlignment(Qt::AlignCenter);
    rightLayout->addWidget(rightTitle_);
    
    rightDisplay_ = new QLabel(parent);
    rightDisplay_->setFixedSize(DISPLAY_W, DISPLAY_H);
    rightDisplay_->setStyleSheet("QLabel { background-color: black; }");
    rightDisplay_->setAlignment(Qt::AlignCenter);
    rightLayout->addWidget(rightDisplay_);
    
    layout->addLayout(rightLayout);
    layout->addStretch();
    
    mainLayout->addLayout(layout);
}

void MainWindow::createStatusBar()
{
    statusLabel_ = new QLabel("No video loaded", this);
    statusBar()->addWidget(statusLabel_);
}

void MainWindow::openFile()
{
    QString filename = QFileDialog::getOpenFileName(
        this, "Open Video File", "",
        "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*.*)"
    );
    
    if (filename.isEmpty()) {
        return;
    }
    
    releaseVideo();
    
    cap_.open(filename.toStdString());
    
    if (!cap_.isOpened()) {
        QMessageBox::critical(this, "Error", 
            QString("Failed to open: %1").arg(QFileInfo(filename).fileName()));
        return;
    }
    
    // Get video properties
    totalFrames_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_COUNT));
    fps_ = cap_.get(cv::CAP_PROP_FPS);
    if (fps_ <= 0) fps_ = 25.0;
    
    frameIdx_ = 0;
    prevGray_ = cv::Mat();
    
    // Update status
    QString status = QString("Loaded %1 | %2 frames @ %3 fps")
        .arg(QFileInfo(filename).fileName())
        .arg(totalFrames_)
        .arg(fps_, 0, 'f', 2);
    statusLabel_->setText(status);
    
    // Show first frame
    showFrame(0);
}

void MainWindow::togglePlay()
{
    if (!cap_.isOpened()) {
        return;
    }
    
    playing_ = !playing_;
    updatePlayButton();
    
    if (playing_) {
        int interval = static_cast<int>(1000.0 / (fps_ * speed_));
        playbackTimer_->start(interval);
    } else {
        playbackTimer_->stop();
    }
}

void MainWindow::updateFrame()
{
    if (!playing_ || !cap_.isOpened()) {
        return;
    }
    
    // Calculate next frame
    int step = static_cast<int>(speed_);
    if (step < 1) step = 1;
    
    frameIdx_ += step;
    
    if (frameIdx_ >= totalFrames_) {
        frameIdx_ = totalFrames_ - 1;
        playing_ = false;
        playbackTimer_->stop();
        updatePlayButton();
        return;
    }
    
    showFrame(frameIdx_);
}

void MainWindow::nextFrame()
{
    if (!cap_.isOpened()) return;
    showFrame(frameIdx_ + 1);
}

void MainWindow::prevFrame()
{
    if (!cap_.isOpened()) return;
    showFrame(frameIdx_ - 1);
}

void MainWindow::faster()
{
    speed_ *= 1.5;
    if (speed_ > 8.0) speed_ = 8.0;
    
    if (playing_) {
        int interval = static_cast<int>(1000.0 / (fps_ * speed_));
        playbackTimer_->setInterval(interval);
    }
    
    QString status = QString("Frame %1/%2 | Mode: %3 | Speed: %4x")
        .arg(frameIdx_ + 1)
        .arg(totalFrames_)
        .arg(flowMode_ == OpticalFlow::DENSE ? "Dense" : "Sparse")
        .arg(speed_, 0, 'f', 1);
    statusLabel_->setText(status);
}

void MainWindow::slower()
{
    speed_ /= 1.5;
    if (speed_ < 0.25) speed_ = 0.25;
    
    if (playing_) {
        int interval = static_cast<int>(1000.0 / (fps_ * speed_));
        playbackTimer_->setInterval(interval);
    }
    
    QString status = QString("Frame %1/%2 | Mode: %3 | Speed: %4x")
        .arg(frameIdx_ + 1)
        .arg(totalFrames_)
        .arg(flowMode_ == OpticalFlow::DENSE ? "Dense" : "Sparse")
        .arg(speed_, 0, 'f', 1);
    statusLabel_->setText(status);
}

void MainWindow::jumpToDialog()
{
    if (!cap_.isOpened()) return;
    
    bool ok;
    int frame = QInputDialog::getInt(
        this, "Jump to Frame", 
        QString("Enter frame number (1-%1):").arg(totalFrames_),
        frameIdx_ + 1, 1, totalFrames_, 1, &ok
    );
    
    if (ok) {
        showFrame(frame - 1);
    }
}

void MainWindow::setFlowMode()
{
    if (denseRadio_->isChecked()) {
        flowMode_ = OpticalFlow::DENSE;
        rightTitle_->setText("Optical Flow - Dense");
    } else {
        flowMode_ = OpticalFlow::SPARSE;
        rightTitle_->setText("Optical Flow - Sparse");
    }
    
    // Recompute flow for current frame
    if (!prevGray_.empty() && !currGray_.empty()) {
        cv::Mat flowVis = opticalFlow_.compute(prevGray_, currGray_, frameBGR_, flowMode_);
        
        if (!flowVis.empty()) {
            cv::Mat flowResized;
            cv::resize(flowVis, flowResized, cv::Size(DISPLAY_W, DISPLAY_H));
            
            QImage qimg(flowResized.data, flowResized.cols, flowResized.rows,
                       flowResized.step, QImage::Format_RGB888);
            rightDisplay_->setPixmap(QPixmap::fromImage(qimg));
        }
    }
}

void MainWindow::showFrame(int idx)
{
    if (!cap_.isOpened()) {
        return;
    }
    
    // Clamp frame index
    idx = std::max(0, std::min(totalFrames_ - 1, idx));
    
    // Set video position
    cap_.set(cv::CAP_PROP_POS_FRAMES, idx);
    
    // Read frame
    cv::Mat frame;
    if (!cap_.read(frame)) {
        statusLabel_->setText("Failed to read frame");
        return;
    }
    
    frameIdx_ = idx;
    frameBGR_ = frame.clone();
    
    // Convert to grayscale
    cv::cvtColor(frameBGR_, currGray_, cv::COLOR_BGR2GRAY);
    
    // Display original frame
    cv::Mat frameRGB;
    cv::cvtColor(frameBGR_, frameRGB, cv::COLOR_BGR2RGB);
    cv::Mat frameResized;
    cv::resize(frameRGB, frameResized, cv::Size(DISPLAY_W, DISPLAY_H));
    
    QImage qimg(frameResized.data, frameResized.cols, frameResized.rows,
               frameResized.step, QImage::Format_RGB888);
    leftDisplay_->setPixmap(QPixmap::fromImage(qimg));
    
    // Compute and display optical flow
    if (!prevGray_.empty()) {
        cv::Mat flowVis = opticalFlow_.compute(prevGray_, currGray_, frameBGR_, flowMode_);
        
        if (!flowVis.empty()) {
            cv::Mat flowResized;
            cv::resize(flowVis, flowResized, cv::Size(DISPLAY_W, DISPLAY_H));
            
            QImage flowImg(flowResized.data, flowResized.cols, flowResized.rows,
                          flowResized.step, QImage::Format_RGB888);
            rightDisplay_->setPixmap(QPixmap::fromImage(flowImg));
        }
    }
    
    // Update previous frame
    prevGray_ = currGray_.clone();
    
    // Update status
    QString status = QString("Frame %1/%2 | Mode: %3 | Speed: %4x")
        .arg(frameIdx_ + 1)
        .arg(totalFrames_)
        .arg(flowMode_ == OpticalFlow::DENSE ? "Dense" : "Sparse")
        .arg(speed_, 0, 'f', 1);
    statusLabel_->setText(status);
}

void MainWindow::releaseVideo()
{
    if (cap_.isOpened()) {
        cap_.release();
    }
    
    playing_ = false;
    playbackTimer_->stop();
    frameIdx_ = 0;
    totalFrames_ = 0;
    prevGray_ = cv::Mat();
    currGray_ = cv::Mat();
    frameBGR_ = cv::Mat();
}

void MainWindow::updatePlayButton()
{
    playButton_->setText(playing_ ? "Pause" : "Play");
}
