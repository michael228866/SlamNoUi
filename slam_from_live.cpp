#include "marker.h"
#include "markerdetector.h"
#include "cameraparameters.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <fstream>
#include <map>
#include <ctime>
#include <curl/curl.h>
#include <iomanip>

using namespace std;
using namespace aruco;
using namespace Eigen;

struct MarkerPose {
    int id;
    cv::Vec3d tvec;
    cv::Vec3d rvec;
};

class KalmanFilter3D {
private:
    cv::KalmanFilter kf;
    cv::Mat state;
    cv::Mat meas;
    double dt;

public:
    KalmanFilter3D(double dt = 1.0 / 30.0) : dt(dt) {
        kf.init(6, 3, 0, CV_64F);
        state = cv::Mat::zeros(6, 1, CV_64F);
        meas = cv::Mat::zeros(3, 1, CV_64F);

        kf.transitionMatrix = (cv::Mat_<double>(6, 6) <<
            1, 0, 0, dt, 0, 0,
            0, 1, 0, 0, dt, 0,
            0, 0, 1, 0, 0, dt,
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1);

        kf.measurementMatrix = cv::Mat::zeros(3, 6, CV_64F);
        for (int i = 0; i < 3; ++i)
            kf.measurementMatrix.at<double>(i, i) = 1.0;

        cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-4));
        cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-2));
        cv::setIdentity(kf.errorCovPost, cv::Scalar::all(0.1));
    }

    cv::Vec3d update(const cv::Vec3d& measurement) {
        kf.predict();
        for (int i = 0; i < 3; ++i)
            meas.at<double>(i) = measurement[i];
        kf.correct(meas);
        for (int i = 0; i < 3; ++i)
            state.at<double>(i) = kf.statePost.at<double>(i);
        return cv::Vec3d(state.at<double>(0), state.at<double>(1), state.at<double>(2));
    }
};

bool loadCameraParams(const string& filename, CameraParameters& camParams) {
    try {
        camParams.readFromXMLFile(filename);
        return camParams.isValid();
    } catch (...) {
        return false;
    }
}

#include <Eigen/Dense>
using namespace Eigen;

map<int, MarkerPose> loadMap(const string& filename, float markerSize) {
    map<int, MarkerPose> poses;
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "❌ 無法開啟地圖檔案：" << filename << endl;
        return poses;
    }

    cv::FileNode markers = fs["aruco_bc_markers"];
    if (markers.empty() || !markers.isSeq()) {
        cerr << "❌ 地圖格式錯誤：缺少 'aruco_bc_markers'" << endl;
        return poses;
    }

    float half = markerSize / 2.0f;
    vector<Vector3d> objPts = {
        {-half,  half, 0},
        { half,  half, 0},
        { half, -half, 0},
        {-half, -half, 0}
    };

    for (const auto& m : markers) {
        int id = (int)m["id"];
        cv::FileNode cornersNode = m["corners"];
        if (cornersNode.size() != 4) continue;

        vector<Vector3d> worldPts;
        for (int i = 0; i < 4; ++i) {
            cv::Vec3f pt;
            cornersNode[i] >> pt;
            worldPts.emplace_back(pt[0], pt[1], pt[2]);
        }

        // 計算中心
        Vector3d center_obj = Vector3d::Zero();
        Vector3d center_world = Vector3d::Zero();
        for (int i = 0; i < 4; ++i) {
            center_obj += objPts[i];
            center_world += worldPts[i];
        }
        center_obj /= 4.0;
        center_world /= 4.0;

        // 去中心化
        MatrixXd X(3, 4), Y(3, 4);
        for (int i = 0; i < 4; ++i) {
            X.col(i) = objPts[i] - center_obj;
            Y.col(i) = worldPts[i] - center_world;
        }

        // SVD 求旋轉
        Matrix3d H = X * Y.transpose();
        JacobiSVD<Matrix3d> svd(H, ComputeFullU | ComputeFullV);
        Matrix3d R = svd.matrixV() * svd.matrixU().transpose();

        // 檢查是否翻轉（避免反射）
        if (R.determinant() < 0) {
            Matrix3d V = svd.matrixV();
            V.col(2) *= -1;
            R = V * svd.matrixU().transpose();
        }

        Vector3d t = center_world - R * center_obj;

        // 儲存結果
        MarkerPose mp;
        mp.id = id;

        // 轉為 OpenCV Vec3d
        cv::Rodrigues((cv::Mat_<double>(3, 3) << 
            R(0, 0), R(0, 1), R(0, 2),
            R(1, 0), R(1, 1), R(1, 2),
            R(2, 0), R(2, 1), R(2, 2)
        ), mp.rvec);
        mp.tvec = cv::Vec3d(t(0), t(1), t(2));

        poses[mp.id] = mp;
    }

    return poses;
}


int main(int argc, char** argv) {
    if (argc < 4) {
        cout << "Usage: slam_from_live <map.yml> <camera.yml> <marker_size_m>" << endl;
        return -1;
    }

    string mapPath = argv[1];
    string camPath = argv[2];
    float markerSize = stof(argv[3]);

    ofstream logFile("pose_log.txt", ios::app);
    if (!logFile.is_open()) {
        cerr << "❌ 無法打開 pose_log.txt 進行寫入" << endl;
        return -1;
    }

    CameraParameters camParams;
    if (!loadCameraParams(camPath, camParams)) {
        cerr << "Cannot load camera parameters from: " << camPath << endl;
        return -1;
    }

    auto map = loadMap(mapPath, markerSize);
    if (map.empty()) {
        cerr << "Cannot load marker map from: " << mapPath << endl;
        return -1;
    }

    cv::VideoCapture cap("libcamerasrc ! videoconvert ! appsink", cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        cerr << "❌ 無法開啟攝影機（libcamerasrc）" << endl;
        return -1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0 || fps > 120) fps = 30.0;
    KalmanFilter3D kalman_filter(1.0 / fps);

    MarkerDetector detector;
    detector.setDictionary("ARUCO_MIP_36h12");
    aruco::MarkerDetector::Params& params = detector.getParameters();
    params.cornerRefinementM = aruco::CORNER_SUBPIX; 
    params.error_correction_rate = 0.6; 

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        vector<Marker> detectedMarkers;
        detector.detect(frame, detectedMarkers, camParams, markerSize);

        vector<cv::Mat> poses;

        for (const auto& marker : detectedMarkers) {
            cout << "\033[1;34m✔ 偵測到 Marker ID:\033[0m " << marker.id << endl;
            logFile << "偵測到 Marker ID: " << marker.id << endl;

            auto it = map.find(marker.id);
            if (it == map.end()) {
                cout << "⚠️ Marker ID " << marker.id << " 不在地圖中，略過。\n";
                continue;
            }

            cv::Mat rvec_marker, tvec_marker;
            cv::Rodrigues(cv::Mat(it->second.rvec), rvec_marker);
            tvec_marker = cv::Mat(it->second.tvec).clone();

            cv::Mat rvec_relative = marker.Rvec.clone();
            cv::Mat tvec_relative = marker.Tvec.clone();

            cv::Mat R_marker_to_cam;
            cv::Rodrigues(rvec_relative, R_marker_to_cam);

            cv::Mat T_marker_to_cam = cv::Mat::eye(4, 4, CV_64F);
            R_marker_to_cam.copyTo(T_marker_to_cam(cv::Rect(0, 0, 3, 3)));
            T_marker_to_cam.at<double>(0, 3) = tvec_relative.at<float>(0);
            T_marker_to_cam.at<double>(1, 3) = tvec_relative.at<float>(1);
            T_marker_to_cam.at<double>(2, 3) = tvec_relative.at<float>(2);

            cv::Mat R_map_marker;
            cv::Rodrigues(cv::Mat(it->second.rvec), R_map_marker);
            cv::Mat T_map_marker = cv::Mat::eye(4, 4, CV_64F);
            R_map_marker.copyTo(T_map_marker(cv::Rect(0, 0, 3, 3)));
            T_map_marker.at<double>(0, 3) = tvec_marker.at<double>(0);
            T_map_marker.at<double>(1, 3) = tvec_marker.at<double>(1);
            T_map_marker.at<double>(2, 3) = tvec_marker.at<double>(2);

            cv::Mat T_map_to_cam = T_marker_to_cam * T_map_marker.inv();  // 先反轉 marker->map 得到 map->marker
            cv::Mat cam_pos = T_map_to_cam(cv::Rect(3, 0, 1, 3)).clone();
            cv::Mat marker_pos = cv::Mat(it->second.tvec).clone();
            cv::Mat dir_to_marker = marker_pos - cam_pos;
            dir_to_marker /= cv::norm(dir_to_marker);

            cv::Mat R_map_to_cam = T_map_to_cam(cv::Rect(0, 0, 3, 3));
            cv::Mat cam_z = R_map_to_cam.t() * (cv::Mat_<double>(3, 1) << 0, 0, 1);  // 相機Z軸在地圖座標下的方向
            cam_z /= cv::norm(cam_z);            

            double dot_product = cam_z.dot(dir_to_marker);
            dot_product = std::max(-1.0, std::min(1.0, dot_product));
            double angle_rad = acos(dot_product);
            double angle_deg = angle_rad * 180.0 / CV_PI;

            if (angle_deg > 60) {
                cout << "角度" << angle_deg << " 超過60度所以，略過。\n";
                continue;
            }
            
            vector<cv::Point2f> reprojected;
            cv::projectPoints(marker.get3DPoints(markerSize), marker.Rvec, marker.Tvec,
                            camParams.CameraMatrix, camParams.Distorsion, reprojected);

            double error = 0.0;
            for (size_t i = 0; i < 4; ++i) {
                error += cv::norm(marker[i] - reprojected[i]);
            }
            if (error > 2.0) {
                cerr << "❌ 標記 " << marker.id << " 重投影誤差過大: " << error << endl;
                continue;
            } else {
                cout << "✅ 標記 " << marker.id << " 重投影誤差: " << error << endl;
            }
            

            cout << "Estimated camera pose (map -> camera):\n" << T_map_to_cam << endl;
            logFile << "Estimated camera pose (map -> camera):\n" << T_map_to_cam << endl;
            poses.push_back(T_map_to_cam);
        }

        if (!poses.empty()) {
            cv::Vec3d avg_t(0, 0, 0);
            vector<Quaterniond> quats;

            for (const auto& T : poses) {
                cv::Mat T_inv = T.inv();  // camera -> map
                avg_t += cv::Vec3d(T_inv.at<double>(0, 3), T_inv.at<double>(1, 3), T_inv.at<double>(2, 3));

                Matrix3d R;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        R(i, j) = T.at<double>(i, j);

                Quaterniond q(R);
                if (q.w() < 0) q.coeffs() *= -1;
                quats.push_back(q);
            }
            avg_t *= (1.0 / poses.size());

            cv::Vec3d filtered_t = kalman_filter.update(avg_t);

            Quaterniond q_avg(0, 0, 0, 0);
            for (const auto& q : quats)
                q_avg.coeffs() += q.coeffs();
            q_avg.normalize();

            Matrix3d R_avg = q_avg.toRotationMatrix();

            cv::Mat T_avg = cv::Mat::eye(4, 4, CV_64F);
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    T_avg.at<double>(i, j) = R_avg(i, j);
            T_avg.at<double>(0, 3) = filtered_t[0];
            T_avg.at<double>(1, 3) = filtered_t[1];
            T_avg.at<double>(2, 3) = filtered_t[2];

            cv::Mat R = T_avg(cv::Rect(0, 0, 3, 3));
            cv::Mat R_cam_to_map = R.t();  // 相機 → 地圖
            double dx = R_cam_to_map.at<double>(0, 2);
            double dz = R_cam_to_map.at<double>(2, 2);
            double yaw_rad = std::atan2(dx, dz);
            double yaw_deg = yaw_rad * 180.0 / CV_PI;

            cout << "\033[1;32m平均相機位置:\033[0m " << filtered_t << endl;
            cout << "平均相機姿態矩陣 (map -> camera):\n" << T_avg << endl;

            logFile << "濾波後相機位置: [" << filtered_t[0] << ", " << filtered_t[1] << ", " << filtered_t[2] << "]\n";
            logFile << "平均相機姿態矩陣 (map -> camera):\n" << T_avg << endl;
            std::time_t now_c = std::time(nullptr);
            logFile << "現在時間: " << std::ctime(&now_c) << endl;

            CURL* curl = curl_easy_init();
            if (curl) {
                std::stringstream json;
                json << std::fixed << std::setprecision(6);
                json << R"({"x": )" << filtered_t[0]
                     << R"(, "y": )" << filtered_t[1]
                     << R"(, "z": )" << filtered_t[2]
                     << R"(, "yaw": )" << yaw_deg << "}";

                std::string json_str = json.str();
                std::cout << "\U0001f4e4 Sent JSON: " << json_str << std::endl;

                struct curl_slist* headers = nullptr;
                headers = curl_slist_append(headers, "Content-Type: application/json");

                curl_easy_setopt(curl, CURLOPT_URL, "http://192.168.9.96:5000/position");
                curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
                curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_str.c_str());
                curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, json_str.length());
                curl_easy_setopt(curl, CURLOPT_POST, 1L);

                CURLcode res = curl_easy_perform(curl);
                if (res != CURLE_OK) {
                    std::cerr << "❌ Failed to send: " << curl_easy_strerror(res) << std::endl;
                } else {
                    std::cout << "✅ Position sent to server.\n";
                }

                curl_slist_free_all(headers);
                curl_easy_cleanup(curl);
            }
        }
    }

    logFile.close();
    return 0;
}