#include "marker.h"
#include "markerdetector.h"
#include "cameraparameters.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>
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

bool loadCameraParams(const string& filename, CameraParameters& camParams) {
    try {
        camParams.readFromXMLFile(filename);
        return camParams.isValid();
    } catch (...) {
        return false;
    }
}

map<int, MarkerPose> loadMap(const string& filename) {
    map<int, MarkerPose> poses;
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) return poses;

    cv::FileNode markers = fs["aruco_bc_markers"];
    if (markers.type() != cv::FileNode::SEQ) {
        cerr << "地圖格式錯誤：缺少 'aruco_bc_markers'" << endl;
        return poses;
    }

    float markerLength = 0.164f;
    float half = markerLength / 2.0f;

    vector<cv::Point3f> objPts = {
        {-half,  half, 0},
        { half,  half, 0},
        { half, -half, 0},
        {-half, -half, 0}
    };

    for (const auto& m : markers) {
        MarkerPose mp;
        mp.id = (int)m["id"];

        cv::FileNode cornersNode = m["corners"];
        if (cornersNode.size() != 4) continue;

        vector<cv::Point3f> realPts;
        for (int i = 0; i < 4; ++i) {
            cv::Vec3f pt;
            cornersNode[i] >> pt;
            realPts.emplace_back(pt);
        }

        cv::Mat affine;
        cv::estimateAffine3D(objPts, realPts, affine, cv::noArray());

        // 將 affine (3x4) 轉換為 R, t
        cv::Mat R = affine(cv::Rect(0, 0, 3, 3)).clone(); // 3x3
        cv::Mat t = affine(cv::Rect(3, 0, 1, 3)).clone(); // 3x1

        cv::Rodrigues(R, mp.rvec);
        mp.tvec = cv::Vec3d(t.at<double>(0), t.at<double>(1), t.at<double>(2));

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

    auto map = loadMap(mapPath);
    if (map.empty()) {
        cerr << "Cannot load marker map from: " << mapPath << endl;
        return -1;
    }

    cv::VideoCapture cap("libcamerasrc ! videoconvert ! appsink", cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        cerr << "❌ 無法開啟攝影機（libcamerasrc）" << endl;
        return -1;
    }

    cout << "實際解析度: " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << " x " << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << endl;
    MarkerDetector detector;
    detector.setDictionary("ARUCO_MIP_36h12");

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

            cv::Mat T_map_to_cam = T_map_marker * T_marker_to_cam;

            // 相機位置（map座標系）
            cv::Mat cam_pos = T_map_to_cam(cv::Rect(3, 0, 1, 3)).clone(); // 3x1

            // marker 在地圖的位置（地圖座標）
            cv::Mat marker_pos = cv::Mat(it->second.tvec).clone(); // 3x1

            // 相機指向 marker 的向量（從相機指到 marker）
            cv::Mat dir_to_marker = marker_pos - cam_pos;
            dir_to_marker /= cv::norm(dir_to_marker);  // 單位化

            // 相機的 Z 軸（forward）方向，在 map 中的表示：T_map_to_cam 的第3欄
            cv::Mat cam_z = T_map_to_cam(cv::Rect(2, 0, 1, 3)).clone(); // 3x1
            cam_z /= cv::norm(cam_z);  // 單位向量

            // 計算夾角（rad 再轉 deg）
            double dot_product = cam_z.dot(dir_to_marker);
            dot_product = std::max(-1.0, std::min(1.0, dot_product));
            double angle_rad = acos(dot_product);
            double angle_deg = angle_rad * 180.0 / CV_PI;

            if (angle_deg>60){
                cout << "角度" << angle_deg << " 超過60度所以，略過。\n";
                continue;
            }

            cout << "Estimated camera pose (map -> camera):\n" << T_map_to_cam << endl;
            logFile << "Estimated camera pose (map -> camera):\n" << T_map_to_cam << endl;
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

            Quaterniond q_avg(0, 0, 0, 0);
            for (const auto& q : quats)
                q_avg.coeffs() += q.coeffs();
            q_avg.normalize();

            Matrix3d R_avg = q_avg.toRotationMatrix();

            cv::Mat T_avg = cv::Mat::eye(4, 4, CV_64F);
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    T_avg.at<double>(i, j) = R_avg(i, j);
            T_avg.at<double>(0, 3) = avg_t[0];
            T_avg.at<double>(1, 3) = avg_t[1];
            T_avg.at<double>(2, 3) = avg_t[2];

			double dx = T_avg.at<double>(0, 2);  // Z 軸方向向量的 x 分量
			double dz = T_avg.at<double>(2, 2);  // Z 軸方向向量的 z 分量

			double yaw_rad = std::atan2(dx, dz);
			double yaw_deg = yaw_rad * 180.0 / CV_PI;
			
            cout << "\033[1;32m平均相機位置:\033[0m " << avg_t << endl;
            cout << "平均相機姿態矩陣 (map -> camera):\n" << T_avg << endl;

            logFile << "平均相機位置: [" << avg_t[0] << ", " << avg_t[1] << ", " << avg_t[2] << "]\n";
            logFile << "平均相機姿態矩陣 (map -> camera):\n" << T_avg << endl;

            std::time_t now_c = std::time(nullptr);
            cout << "現在時間: " << std::ctime(&now_c);
            logFile << "現在時間: " << std::ctime(&now_c) << endl;
			
			CURL* curl = curl_easy_init();
			if (curl) {
				std::stringstream json;
				json << std::fixed << std::setprecision(6);
				json << R"({"x": )" << avg_t[0]
					 << R"(, "y": )" << avg_t[1]
					 << R"(, "z": )" << avg_t[2]
					 << R"(, "yaw": )" << yaw_deg << "}";


				std::string json_str = json.str();
				std::cout << "📤 Sent JSON: " << json_str << std::endl;

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
