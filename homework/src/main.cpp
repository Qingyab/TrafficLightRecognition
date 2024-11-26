#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // 打开视频流
    string VedioPath="C:/Users/22132/Desktop/color-main/video/random.avi";
    VideoCapture cap(VedioPath);
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open the camera" << endl;
        return -1;
    }

    Mat frame, fgMask, hsv, redMask, greenMask, yellowMask;
    Ptr<BackgroundSubtractor> pBackSub;
    pBackSub = createBackgroundSubtractorMOG2();

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            cerr << "Error: Blank frame grabbed" << endl;
            break;
        }

        pBackSub->apply(frame, fgMask);

        cvtColor(frame, hsv, COLOR_BGR2HSV);

        //掩模
        Mat redMask1, redMask2;
        inRange(hsv, Scalar(0, 150, 150), Scalar(10, 255, 255), redMask1);
        inRange(hsv, Scalar(160, 150, 150), Scalar(180, 255, 255), redMask2);
        bitwise_or(redMask1, redMask2, redMask);

        inRange(hsv, Scalar(40, 110, 110), Scalar(80, 255, 255), greenMask);

        inRange(hsv, Scalar(15, 130, 130), Scalar(35, 255, 255), yellowMask);

        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        morphologyEx(redMask, redMask, MORPH_CLOSE, kernel);
        morphologyEx(greenMask, greenMask, MORPH_CLOSE, kernel);
        morphologyEx(yellowMask, yellowMask, MORPH_CLOSE, kernel);

        // 标志变量
        bool redDetected = false;
        bool greenDetected = false;
        bool yellowDetected = false;

        vector<vector<Point>> contours;

        // 检测
        findContours(redMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        for (const auto& contour : contours) {
            double area = contourArea(contour);
            if (area > 100) {
                Point2f center;
                float radius;
                minEnclosingCircle(contour, center, radius);
                if (radius > 10 && radius < 50) {
                    double redArea=3.141*radius*radius;
                    if(area>(0.55)*redArea){
                        circle(frame, center, (int)radius, Scalar(0, 0, 255), 3);
                        redDetected = true;
                    }
                }
            }
        }

        // 绿灯检测
        findContours(greenMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        for (const auto& contour : contours) {
            double area = contourArea(contour);
            if (area > 100) {
                Point2f center;
                float radius;
                minEnclosingCircle(contour, center, radius);
                if (radius > 10 && radius < 50) {
                    double greenArea=3.141*radius*radius;
                    if(area>(0.55)*greenArea){
                        circle(frame, center, (int)radius, Scalar(0, 255, 0), 3);
                        greenDetected = true;
                    }
                }
            }
        }

        // 黄灯检测
        findContours(yellowMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        for (const auto& contour : contours) {
            double area = contourArea(contour);
            if (area > 100) {
                Point2f center;
                float radius;
                minEnclosingCircle(contour, center, radius);
                if (radius > 10 && radius < 50) {
                    double yellowArea=3.141*radius*radius;
                    if(area>(0.55)*yellowArea){
                        circle(frame, center, (int)radius, Scalar(0, 255, 255), 3);
                        yellowDetected = true;
                    }
                }
            }
        }

        // 显示检测结果
        string detectionResult;
        if (redDetected || greenDetected || yellowDetected) {
            if (redDetected) detectionResult += "RED ";
            if (greenDetected) detectionResult += "GREEN ";
            if (yellowDetected) detectionResult += "YELLOW ";
        } else {
            detectionResult = "NOT";
        }

        // 在左上角显示结果
        putText(frame, detectionResult, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);

        // 显示结果
        imshow("Traffic Light Recognition", frame);
        imshow("ForeGround Mask", fgMask);

        char c = (char)waitKey(25);
        if (c == 27) // 按下ESC退出
            break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
