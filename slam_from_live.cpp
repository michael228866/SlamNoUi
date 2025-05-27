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
#include <iomanip>
#include <ixwebsocket/IXWebSocket.h>

ix::WebSocket websocket;


using namespace std;
using namespace aruco;
using namespace Eigen;

struct MarkerPose {
    int id;
    cv::Vec3d tvec;
    cv::Vec3d rvec;
};

// class KalmanFilter3D {
// private:
//     cv::KalmanFilter kf;
//     cv::Mat state;
//     cv::Mat meas;
//     double dt;

// public:
//     KalmanFilter3D(double dt = 1.0 / 30.0) : dt(dt) {
//         kf.init(6, 3, 0, CV_64F);
//         state = cv::Mat::zeros(6, 1, CV_64F);
//         meas = cv::Mat::zeros(3, 1, CV_64F);

//         kf.transitionMatrix = (cv::Mat_<double>(6, 6) <<
//             1, 0, 0, dt, 0, 0,
//             0, 1, 0, 0, dt, 0,
//             0, 0, 1, 0, 0, dt,
//             0, 0, 0, 1, 0, 0,
//             0, 0, 0, 0, 1, 0,
//             0, 0, 0, 0, 0, 1);

//         kf.measurementMatrix = cv::Mat::zeros(3, 6, CV_64F);
//         for (int i = 0; i < 3; ++i)
//             kf.measurementMatrix.at<double>(i, i) = 1.0;

//         cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-3));
//         cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
//         cv::setIdentity(kf.errorCovPost, cv::Scalar::all(0.1));
//     }

//     cv::Vec3d update(const cv::Vec3d& measurement) {
//         kf.predict();
//         for (int i = 0; i < 3; ++i)
//             meas.at<double>(i) = measurement[i];
//         kf.correct(meas);
//         for (int i = 0; i < 3; ++i)
//             state.at<double>(i) = kf.statePost.at<double>(i);
//         return cv::Vec3d(state.at<double>(0), state.at<double>(1), state.at<double>(2));
//     }

//     cv::Vec3d predictOnly() {
//     kf.predict();
//     for (int i = 0; i < 3; ++i)
//         state.at<double>(i) = kf.statePre.at<double>(i);
//     return cv::Vec3d(state.at<double>(0), state.at<double>(1), state.at<double>(2));
//     }
// };

bool loadCameraParams(const string& filename, CameraParameters& camParams) {
    try {
        camParams.readFromXMLFile(filename);
        return camParams.isValid();
    } catch (...) {
        return false;
    }
}

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
        cv::Mat R_cv = (cv::Mat_<double>(3, 3) <<
            R(0, 0), R(0, 1), R(0, 2),
            R(1, 0), R(1, 1), R(1, 2),
            R(2, 0), R(2, 1), R(2, 2));
        cv::Rodrigues(R_cv, mp.rvec);
        mp.tvec = cv::Vec3d(t(0), t(1), t(2));

        poses[mp.id] = mp;
    }

    return poses;
}

const int SLIDING_WINDOW = 5;
std::deque<cv::Vec3d> recent_positions;

//可信度
double calculateConfidence(double reprojection_error, double angle_deg, double distance,float markerSize) {
    // 設定權重與限制
    const double max_error = 4.0;      // 誤差上限
    const double max_angle = 60.0;     // 角度上限（越小越好）
    double scale_factor = 12.0;

    const double max_distance = markerSize*scale_factor;   // 距離上限（依場景調整）

    // 各項評分 (0~1)，越接近 1 越好
    double score_err = std::max(0.0, 1.0 - (reprojection_error / max_error));
    double score_angle = std::max(0.0, 1.0 - (angle_deg / max_angle));
    double score_dist = std::max(0.0, 1.0 - (distance / max_distance));

    // 加權平均 (可調整比重)
    double confidence = (score_err * 0.4) + (score_angle * 0.4) + (score_dist * 0.2);
    return confidence;
}


int main(int argc, char** argv) {

   cout << "🚀 程式啟動" << endl;

    if (argc < 4) {
        cout << "❗ 參數不足" << endl;
        return -1;
    }

    cout << "📁 載入參數..." << endl;
    string mapPath = argv[1];
    string camPath = argv[2];
    float markerSize = stof(argv[3]);

    cout << "📷 嘗試載入相機參數..." << endl;
    CameraParameters camParams;
    if (!loadCameraParams(camPath, camParams)) {
        cerr << "❌ 相機參數讀取失敗：" << camPath << endl;
        return -1;
    }
    cout << "✅ 相機參數讀取成功" << endl;

    cout << "🗺️ 嘗試載入 marker map..." << endl;
    auto map = loadMap(mapPath, markerSize);
    if (map.empty()) {
        cerr << "❌ 地圖讀取失敗：" << mapPath << endl;
        return -1;
    }
    cout << "✅ 地圖載入完成，共有 " << map.size() << " 個 marker" << endl;

    cout << "📷 嘗試開啟攝影機..." << endl;
    cv::VideoCapture cap("libcamerasrc ! videoconvert ! appsink", cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        cerr << "❌ 攝影機打不開（GStreamer pipeline）" << endl;
        return -1;
    }
    cout << "✅ 攝影機開啟成功" << endl;


    // 設定 WebSocket URL
    websocket.setUrl("ws://192.168.9.96:5000?type=vr_headset&clientId=vr001");

    // 設定訊息接收 callback（這是你需要的最基本功能）
    websocket.setOnMessageCallback([](const ix::WebSocketMessagePtr& msg) {
        if (msg->type == ix::WebSocketMessageType::Message) {
            std::cout << "📩 收到訊息：" << msg->str << std::endl;
        } else if (msg->type == ix::WebSocketMessageType::Open) {
            std::cout << "✅ WebSocket 已連線" << std::endl;
        } else if (msg->type == ix::WebSocketMessageType::Error) {
            std::cerr << "❌ WebSocket 錯誤：" << msg->errorInfo.reason << std::endl;
        } else if (msg->type == ix::WebSocketMessageType::Close) {
            std::cout << "❌ WebSocket 關閉，原因: " << msg->closeInfo.reason << std::endl;
        }
    });

    websocket.start();

    // double fps = cap.get(cv::CAP_PROP_FPS);
    // KalmanFilter3D kalman_filter(1.0 / fps );

    MarkerDetector detector;
    detector.setDictionary("ARUCO_MIP_36h12");
    detector.getParameters().setCornerRefinementMethod(aruco::CornerRefinementMethod::CORNER_SUBPIX);

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        vector<Marker> detectedMarkers;
        detector.detect(frame, detectedMarkers, camParams, markerSize);

        vector<cv::Mat> poses;
        
        for (const auto& marker : detectedMarkers) {
            cout << "\033[1;34m✔ 偵測到 Marker ID:\033[0m " << marker.id << endl;

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

            cv::Mat T_map_to_cam = T_map_marker * T_marker_to_cam;
            cv::Mat cam_pos = T_map_to_cam(cv::Rect(3, 0, 1, 3)).clone();
            cv::Mat marker_pos = cv::Mat(it->second.tvec).clone();
            cv::Mat dir_to_marker = marker_pos - cam_pos;
            dir_to_marker /= cv::norm(dir_to_marker);

            cv::Mat cam_z = T_map_to_cam(cv::Rect(2, 0, 1, 3)).clone();
            cam_z /= cv::norm(cam_z);

            double dot_product = cam_z.dot(dir_to_marker);
            dot_product = std::max(-1.0, std::min(1.0, dot_product));
            double angle_rad = acos(dot_product);
            double angle_deg = angle_rad * 180.0 / CV_PI;

            if (angle_deg > 50) {
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
            if (error > 4.0) {
                cerr << "❌ 標記 " << marker.id << " 重投影誤差過大: " << error << endl;
                continue;
            } else {
                cout << "✅ 標記 " << marker.id << " 重投影誤差: " << error << endl;
            }

            double distance = cv::norm(marker.Tvec);
            double confidence = calculateConfidence(error, angle_deg, distance,markerSize);
            cout << "🔍 信心值 (Confidence): " << confidence << endl;

            // 可選：加入信心值過低則跳過
            if (confidence < 0.5) {
                cout << "⚠️ 可信度太低 (" << confidence << ")，略過此 marker。\n";
                continue;
            }


            cout << "Estimated camera pose (map -> camera):\n" << T_map_to_cam << endl;
            poses.push_back(T_map_to_cam);
        }



        if (!poses.empty()) {
            cv::Vec3d avg_t(0, 0, 0);
            vector<Quaterniond> quats;

            for (const auto& T : poses) {
                avg_t += cv::Vec3d(T.at<double>(0, 3), T.at<double>(1, 3), T.at<double>(2, 3));

                Matrix3d R;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        R(i, j) = T.at<double>(i, j);

                Quaterniond q(R);
                if (q.w() < 0) q.coeffs() *= -1;
                quats.push_back(q);
            }
            avg_t *= (1.0 / poses.size());

            // ➤ 加入進滑動視窗
            recent_positions.push_back(avg_t);
            if (recent_positions.size() > SLIDING_WINDOW)
                recent_positions.pop_front();

            // ➤ 計算滑動平均
            cv::Vec3d smooth_t(0, 0, 0);
            int N = recent_positions.size();
            double total_weight = 0.0;

            for (int i = 0; i < N; ++i) {
                double weight = (i + 1);  // 權重線性成長：1, 2, ..., N
                total_weight += weight;
                smooth_t += recent_positions[i] * weight;
            }
            smooth_t *= (1.0 / total_weight);

            // ➤ 再進 Kalman 濾波
            // cv::Vec3d filtered_t = kalman_filter.update(smooth_t);

            cv::Vec3d filtered_t = smooth_t;

            
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

            // 提取相機姿態三軸方向（世界坐標中）
            cv::Vec3d x_axis(T_avg.at<double>(0, 0), T_avg.at<double>(1, 0), T_avg.at<double>(2, 0));
            cv::Vec3d y_axis(T_avg.at<double>(0, 1), T_avg.at<double>(1, 1), T_avg.at<double>(2, 1));
            cv::Vec3d z_axis(T_avg.at<double>(0, 2), T_avg.at<double>(1, 2), T_avg.at<double>(2, 2));

            x_axis /= cv::norm(x_axis);
            y_axis /= cv::norm(y_axis);
            z_axis /= cv::norm(z_axis);

            cout << "🧭 X 軸方向 (右): " << x_axis << endl;
            cout << "🧭 Y 軸方向 (上): " << y_axis << endl;
            cout << "🧭 Z 軸方向 (前): " << z_axis << endl;


            cout << "\033[1;32m平均相機位置:\033[0m " << filtered_t << endl;
            cout << "平均相機姿態矩陣 (map -> camera):\n" << T_avg << endl;


            std::stringstream json;
            json << std::fixed << std::setprecision(6);
            json << R"({"type": "transform_update", "headsetId": "1", "pos": [)"
                << filtered_t[0] << ", " << filtered_t[1] << ", " << filtered_t[2]
                << R"(], "orient": [)"
                << z_axis[0] << ", " << z_axis[1] << ", " << z_axis[2] << "]}";

            std::string json_str = json.str();
            std::cout << "📤 Sent JSON: " << json_str << std::endl;

            websocket.send(json_str);
            
        }else {
            // 沒有偵測到 marker，也做 Kalman 預測
            // cv::Vec3d filtered_t = kalman_filter.predictOnly();
        }
    }

    return 0;


}