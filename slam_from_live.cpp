#include "marker.h" // from libaruco
#include "markerdetector.h" // from libaruco
#include "cameraparameters.h" // from libaruco
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp> // For cv::Rodrigues, cv::solvePnP
#include <opencv2/video/tracking.hpp> // For cv::KalmanFilter
#include <Eigen/Dense> // For Quaternion, Matrix3d
#include <Eigen/Geometry> // For eulerAngles
#include <iostream>
#include <fstream>
#include <map>
#include <ctime>
#include <curl/curl.h> // For HTTP POST
#include <iomanip> // For std::fixed, std::setprecision

using namespace std;
using namespace aruco; // This is the libaruco namespace
using namespace Eigen;

// MarkerPose 結構體用於儲存地圖中每個標記在世界坐標系中的姿態
struct MarkerPose {
    int id;
    cv::Vec3d tvec; // 平移向量 (世界坐標系原點到標記中心)
    cv::Vec3d rvec; // 旋轉向量 (世界坐標系到標記局部坐標系)
};

class KalmanFilter3D {
private:
    cv::KalmanFilter kf;
    cv::Mat state; // 狀態向量 [x, y, z, vx, vy, vz]'
    cv::Mat meas;  // 測量向量 [x, y, z]'
    double dt;

public:
    KalmanFilter3D(double dt = 1.0 / 30.0) : dt(dt) {
        // 6 狀態變量 (x, y, z, vx, vy, vz), 3 測量變量 (x, y, z)
        kf.init(6, 3, 0, CV_64F);
        state = cv::Mat::zeros(6, 1, CV_64F);
        meas = cv::Mat::zeros(3, 1, CV_64F);

        // 狀態轉移矩陣 A
        // x = x_prev + vx * dt
        // vx = vx_prev
        kf.transitionMatrix = (cv::Mat_<double>(6, 6) <<
            1, 0, 0, dt, 0, 0,
            0, 1, 0, 0, dt, 0,
            0, 0, 1, 0, 0, dt,
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1);

        // 測量矩陣 H
        // 我們直接測量位置 x, y, z
        kf.measurementMatrix = (cv::Mat_<double>(3, 6) <<
            1, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0);

        // 過程噪聲協方差 Q (模型不確定性)
        // 值越大，濾波器越信任新的測量
        cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-4)); // 較小的過程噪聲，假設模型相對準確

        // 測量噪聲協方差 R (傳感器噪聲)
        // 值越大，濾波器越不信任新的測量
        cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-2)); // 較大的測量噪聲，表示測量可能不夠精確

        // 後驗估計誤差協方差 P (初始不確定性)
        cv::setIdentity(kf.errorCovPost, cv::Scalar::all(0.1)); // 初始誤差，較大值表示對初始狀態不確定
    }

    // 更新濾波器，輸入當前測量值 (相機位置)
    cv::Vec3d update(const cv::Vec3d& measurement) {
        kf.predict(); // 預測下一步的狀態
        for (int i = 0; i < 3; ++i) {
            meas.at<double>(i) = measurement[i]; // 將測量值放入測量矩陣
        }
        kf.correct(meas); // 校正預測狀態，結合測量值
        // 返回校正後的狀態中的位置部分
        return cv::Vec3d(kf.statePost.at<double>(0), kf.statePost.at<double>(1), kf.statePost.at<double>(2));
    }
};

// 載入相機參數 (使用 aruco::CameraParameters)
bool loadCameraParams(const string& filename, CameraParameters& camParams) {
    try {
        camParams.readFromXMLFile(filename);
        return camParams.isValid();
    } catch (const std::exception& e) { // 捕獲更具體的異常
        std::cerr << "載入相機參數時發生錯誤: " << e.what() << std::endl;
        return false;
    } catch (...) { // 捕獲其他未知異常
        std::cerr << "載入相機參數時發生未知錯誤！" << std::endl;
        return false;
    }
}

// 修正後的 loadMap 函數
// 目標是讀取每個 Aruco 標記在世界坐標系中的姿態 (tvec, rvec)
map<int, MarkerPose> loadMap(const string& filename) {
    map<int, MarkerPose> poses;
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "無法打開地圖文件: " << filename << std::endl;
        return poses;
    }

    cv::FileNode markersNode = fs["aruco_bc_markers"];
    if (markersNode.empty() || !markersNode.isSeq()) {
        std::cerr << "地圖格式錯誤：無法找到 'aruco_bc_markers' 節點或其不是序列." << std::endl;
        fs.release();
        return poses;
    }

    float inferred_marker_size = 0.0;
    if (markersNode.size() > 0) {
        cv::FileNode firstMarkerCornersNode = markersNode[0]["corners"];
        if (!firstMarkerCornersNode.empty() && firstMarkerCornersNode.isSeq() && firstMarkerCornersNode.size() == 4) {
            std::vector<double> corner1_coords, corner2_coords;
            firstMarkerCornersNode[0] >> corner1_coords; // top-left
            firstMarkerCornersNode[1] >> corner2_coords; // top-right
            if (corner1_coords.size() == 3 && corner2_coords.size() == 3) {
                // 計算兩個相鄰角點的距離作為邊長
                inferred_marker_size = static_cast<float>(cv::norm(cv::Point3d(corner1_coords[0], corner1_coords[1], corner1_coords[2]) - // **** 修改點 1: Point3d ****
                                                                     cv::Point3d(corner2_coords[0], corner2_coords[1], corner2_coords[2]))); // **** 修改點 1: Point3d ****
            }
        }
    }
    if (inferred_marker_size == 0.0) {
        std::cerr << "無法從地圖中推斷標記尺寸，請檢查地圖檔案格式或確保包含 ID 0 標記。" << std::endl;
        fs.release();
        return poses;
    }

    // 標記在自身局部坐標系中的標準 3D 角點 (假設中心在原點，Z=0 平面)
    float half_marker_size = inferred_marker_size / 2.0f;
    std::vector<cv::Point3d> marker_local_corners = { // **** 修改點 2: Point3d ****
        {-half_marker_size,  half_marker_size, 0},
        { half_marker_size,  half_marker_size, 0},
        { half_marker_size, -half_marker_size, 0},
        {-half_marker_size, -half_marker_size, 0}
    };


    for (cv::FileNodeIterator it = markersNode.begin(); it != markersNode.end(); ++it) {
        MarkerPose mp;
        mp.id = (int)(*it)["id"];

        cv::FileNode cornersNode = (*it)["corners"];
        if (cornersNode.empty() || !cornersNode.isSeq() || cornersNode.size() != 4) {
            std::cerr << "標記 " << mp.id << " 沒有 'corners' 節點或格式不正確." << std::endl;
            continue;
        }

        std::vector<cv::Point3d> world_corners; // **** 修改點 3: Point3d ****
        for (int i = 0; i < 4; ++i) {
            std::vector<double> pt_coords;
            cornersNode[i] >> pt_coords;
            if (pt_coords.size() == 3) {
                world_corners.push_back(cv::Point3d(pt_coords[0], pt_coords[1], pt_coords[2])); // **** 修改點 4: Point3d ****
            } else {
                std::cerr << "標記 " << mp.id << " 的角點座標數量不正確." << std::endl;
                world_corners.clear();
                break;
            }
        }

        if (world_corners.size() == 4) {
            cv::Mat affine_transform; // 4x4 齊次變換矩陣 (3x4 affine matrix)
            cv::estimateAffine3D(marker_local_corners, world_corners, affine_transform, cv::noArray());

            if (affine_transform.empty() || affine_transform.rows != 3 || affine_transform.cols != 4) {
                std::cerr << "無法為標記 " << mp.id << " 估計仿射變換。" << std::endl;
                continue;
            }

            cv::Mat R_mat = affine_transform(cv::Rect(0, 0, 3, 3));
            cv::Mat t_mat = affine_transform(cv::Rect(3, 0, 1, 3));

            cv::Rodrigues(R_mat, mp.rvec);
            mp.tvec = cv::Vec3d(t_mat.at<double>(0, 0), t_mat.at<double>(1, 0), t_mat.at<double>(2, 0)); // 從 Mat 轉換為 Vec3d

            poses[mp.id] = mp;
        }
    }
    fs.release();
    return poses;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        cout << "Usage: slam_from_live <map.yml> <camera.yml> <marker_size_m>" << endl;
        return -1;
    }

    string mapPath = argv[1];
    string camPath = argv[2];
    float markerSize = stof(argv[3]); // 傳入的 markerSize 用於 Aruco 偵測器估計單個標記姿態

    std::vector<cv::Mat> valid_poses;

    ofstream logFile("pose_log.txt", ios::app);
    if (!logFile.is_open()) {
        cerr << "❌ 無法打開 pose_log.txt 進行寫入" << endl;
        return -1;
    }

    CameraParameters camParams; // 使用 aruco::CameraParameters
    if (!loadCameraParams(camPath, camParams)) {
        cerr << "Cannot load camera parameters from: " << camPath << endl;
        return -1;
    }

    // 載入地圖，現在 map 存儲的是每個標記在世界坐標系中的姿態
    auto map_marker_poses = loadMap(mapPath);
    if (map_marker_poses.empty()) {
        cerr << "Cannot load marker map from: " << mapPath << endl;
        return -1;
    }

    // 確保 camParams 的 camMatrix 和 distCoeffs 是 OpenCV 的 Mat 類型
    // aruco::CameraParameters 內部可能已經是 cv::Mat
    // 這裡假設 camParams.CameraMatrix 和 camParams.DistorsionCoeffs 可以直接傳遞給 OpenCV 函式
    if (camParams.CameraMatrix.empty() || camParams.Distorsion.empty()) {
        cerr << "相機參數不完整 (CameraMatrix 或 DistorsionCoeffs 為空)。" << endl;
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

    MarkerDetector detector; // 使用 libaruco 的 MarkerDetector
    detector.setDictionary("ARUCO_MIP_36h12"); // 確保與地圖中的字典一致
    detector.getParameters().setCornerRefinementMethod(aruco::CornerRefinementMethod::CORNER_SUBPIX);

    // 确保相机矩阵和畸变系数是 CV_64F 类型
    if (camParams.CameraMatrix.type() != CV_64F) {
        camParams.CameraMatrix.convertTo(camParams.CameraMatrix, CV_64F);
    }
    if (camParams.Distorsion.type() != CV_64F) {
        camParams.Distorsion.convertTo(camParams.Distorsion, CV_64F);
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        vector<Marker> detectedMarkers;
        // 偵測器會直接計算每個檢測到的標記相對於相機的姿態 (marker.Rvec, marker.Tvec)
        // 這裡的 markerSize 參數用於告知 detector 標記的物理尺寸，以便它能進行姿態估計
        detector.detect(frame, detectedMarkers, camParams, markerSize);

        // 用於 PnP 求解的點集合
        std::vector<cv::Point3f> all_obj_points; // 世界坐標系中的 3D 點
        std::vector<cv::Point2f> all_img_points; // 圖像中的 2D 像素點

        for (const auto& marker : detectedMarkers) {
            cout << "\033[1;34m✔ 偵測到 Marker ID:\033[0m " << marker.id << endl;
            logFile << "偵測到 Marker ID: " << marker.id << endl;

            auto it = map_marker_poses.find(marker.id);
            if (it == map_marker_poses.end()) {
                cout << "⚠️ Marker ID " << marker.id << " 不在地圖中，略過。\n";
                continue;
            }

            // marker.Rvec 和 marker.Tvec 是從**標記局部坐標系**到**相機坐標系**的變換。
            // it->second.rvec 和 it->second.tvec 是從**世界坐標系**到**標記局部坐標系**的變換。

            // Step 1: 獲取相機相對於標記的姿態 (marker.Rvec, marker.Tvec)
            cv::Mat R_marker_to_cam, T_marker_to_cam;
            cv::Rodrigues(marker.Rvec, R_marker_to_cam);
            T_marker_to_cam = cv::Mat(marker.Tvec); // 直接使用 marker.Tvec，確保是 Mat 類型

            // Step 2: 獲取標記相對於世界坐標系的姿態 (it->second.rvec, it->second.tvec)
            cv::Mat R_world_to_marker, T_world_to_marker;
            cv::Rodrigues(it->second.rvec, R_world_to_marker);
            T_world_to_marker = cv::Mat(it->second.tvec); // 直接使用 it->second.tvec

            // 構建齊次變換矩陣
            // T_marker_to_cam_homo (4x4): 從標記局部坐標系到相機坐標系的齊次變換
            cv::Mat T_marker_to_cam_homo = cv::Mat::eye(4, 4, CV_64F);
            R_marker_to_cam.copyTo(T_marker_to_cam_homo(cv::Rect(0, 0, 3, 3)));
            T_marker_to_cam.copyTo(T_marker_to_cam_homo(cv::Rect(3, 0, 1, 3)));

            // T_world_to_marker_homo (4x4): 從世界坐標系到標記局部坐標系的齊次變換
            cv::Mat T_world_to_marker_homo = cv::Mat::eye(4, 4, CV_64F);
            R_world_to_marker.copyTo(T_world_to_marker_homo(cv::Rect(0, 0, 3, 3)));
            T_world_to_marker.copyTo(T_world_to_marker_homo(cv::Rect(3, 0, 1, 3)));

            // 計算 T_world_to_cam (從世界坐標系到相機坐標系的變換)
            // T_world_to_cam = T_marker_to_cam * T_world_to_marker 的逆
            // 或者： T_world_to_cam = ( T_world_to_marker * T_marker_to_cam_inverse )
            // 如果 T_world_to_marker 是 (世界 -> 標記), T_marker_to_cam 是 (標記 -> 相機)
            // 那麼 T_world_to_cam (直接是世界到相機) = T_marker_to_cam * T_world_to_marker_inverse
            // T_marker_to_cam_inverse 是 (相機 -> 標記)
            // 也就是說： T_cam_in_world = T_world_to_marker * T_cam_from_marker_inverse
            // 這裡的 T_cam_from_marker 是 marker.Rvec/Tvec
            // 因此，我們需要計算 T_world_to_marker 變換的逆，然後乘以 T_marker_to_cam
            // 或者更直接的： 攝像機在世界坐標系下的位置 (世界坐標系到攝像機坐標系)
            // T_world_to_cam = T_world_to_marker * T_marker_to_cam_homo
            // 這與您原來的 `T_map_to_cam = T_map_marker * T_marker_to_cam;` 邏輯是一致的，
            // 只是這裡 `T_map_marker` 是 `T_world_to_marker_homo`，`T_marker_to_cam` 是 `T_marker_to_cam_homo`
            cv::Mat T_world_to_cam = T_marker_to_cam_homo * T_world_to_marker_homo;
            // 這裡的 T_world_to_cam 就是我們需要的從世界坐標系到相機坐標系的變換矩陣

            // 從 T_world_to_cam 提取相機位置和姿態
            cv::Mat current_rvec, current_tvec;
            cv::Rodrigues(T_world_to_cam(cv::Rect(0, 0, 3, 3)), current_rvec);
            current_tvec = T_world_to_cam(cv::Rect(3, 0, 1, 3)).clone();

            // 為了後續的姿態平均，我們可以將這些姿勢添加到一個列表中
            all_obj_points.push_back(cv::Point3f(current_tvec.at<double>(0), current_tvec.at<double>(1), current_tvec.at<double>(2)));
            // 將相機的旋轉矩陣轉換為四元數並收集
            Eigen::Matrix3d R_current_eigen;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    R_current_eigen(i, j) = T_world_to_cam.at<double>(i, j);
                }
            }
            // Eigen::Quaterniond q_current(R_current_eigen);
            // quats.push_back(q_current); // 如果有多個姿態需要平均，則放入這裡

            // 這裡的 `angle_deg` 判斷需要重新定義其含義
            // 如果您想過濾掉標記不在視野正前方的情況，可以這樣判斷：
            // 計算相機到標記中心的向量，以及相機坐標系的 Z 軸向量。
            // 然後計算它們的夾角。
            cv::Vec3d marker_center_world = cv::Vec3d(it->second.tvec[0], it->second.tvec[1], it->second.tvec[2]); // 標記中心在世界坐標系
            cv::Vec3d cam_pos_world = cv::Vec3d(current_tvec.at<double>(0), current_tvec.at<double>(1), current_tvec.at<double>(2));

            cv::Vec3d vec_cam_to_marker_world = marker_center_world - cam_pos_world; // 從相機到標記的向量（在世界坐標系中）
            vec_cam_to_marker_world /= cv::norm(vec_cam_to_marker_world); // 歸一化

            cv::Mat R_cam_to_world; // 相機到世界
            cv::transpose(R_marker_to_cam, R_cam_to_world); // 標記到相機的逆就是相機到標記
            cv::Mat cam_z_axis_cam = (cv::Mat_<double>(3,1) << 0, 0, 1); // 相機坐標系的 Z 軸
            cv::Mat cam_z_axis_world_mat = R_cam_to_world * cam_z_axis_cam; // 相機 Z 軸在世界坐標系中的方向

            cv::Vec3d cam_z_axis_world = cv::Vec3d(cam_z_axis_world_mat.at<double>(0), cam_z_axis_world_mat.at<double>(1), cam_z_axis_world_mat.at<double>(2));
            cam_z_axis_world /= cv::norm(cam_z_axis_world); // 歸一化

            double dot_product = cam_z_axis_world.dot(vec_cam_to_marker_world);
            double angle_rad = acos(std::max(-1.0, std::min(1.0, dot_product)));
            double angle_deg = angle_rad * 180.0 / CV_PI;

            if (angle_deg > 60) { // 如果相機不是大致朝向標記，則跳過
                cout << "角度 " << angle_deg << " 度超過60度，可能是標記在視野邊緣，略過。\n";
                continue; // 跳過當前標記的姿態貢獻
            }

            cout << "Estimated camera pose (world -> camera):\n" << T_world_to_cam << endl;
            logFile << "Estimated camera pose (world -> camera):\n" << T_world_to_cam << endl;
            // 如果有多個有效姿態，可以將它們加入一個列表以進行後續平均
            // poses.push_back(T_world_to_cam);
            // 這裡我們直接使用單個標記的結果，或者在循環外進行平均
            // 為了保持與您原代碼的poses邏輯一致，我們將 T_world_to_cam 加入 poses
            // 注意：poses 變量在原代碼中是一個 vector<cv::Mat>，我會繼續使用這個。
            all_obj_points.push_back(cv::Point3f(current_tvec.at<double>(0), current_tvec.at<double>(1), current_tvec.at<double>(2)));
            // 實際上，如果使用 libaruco 並且地圖已經定義了 marker 的世界坐標系姿態，
            // 通常的做法是：
            // 1. marker.Rvec, marker.Tvec (相機相對於標記)
            // 2. map_marker_poses[marker.id].rvec, map_marker_poses[marker.id].tvec (世界相對於標記)
            // 計算 T_world_to_cam = T_marker_to_cam_inverse * T_world_to_marker_inverse
            // T_cam_world = T_marker_cam.inv() * T_marker_world.inv()
            // 這個 `T_map_to_cam = T_marker_to_cam_homo * T_world_to_marker_homo;` 是正確的。

            // 為了簡單起見，我們將每個有效標記的 T_world_to_cam 存儲起來，最後再進行平均
            // 這裡我直接用一個 vector<cv::Mat> 來儲存每個有效標記計算出的 T_world_to_cam
            // 然後在迴圈外進行平均
            static std::vector<cv::Mat> valid_poses; // 靜態變量，用於收集本幀所有有效姿態
            valid_poses.push_back(T_world_to_cam);

        } // end of for (const auto& marker : detectedMarkers)

        if (!valid_poses.empty()) {
            cv::Vec3d avg_t(0, 0, 0);
            vector<Quaterniond> quats;

            for (const auto& T : valid_poses) {
                avg_t += cv::Vec3d(T.at<double>(0, 3), T.at<double>(1, 3), T.at<double>(2, 3));

                Matrix3d R_eigen;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        R_eigen(i, j) = T.at<double>(i, j);

                Quaterniond q(R_eigen);
                // 處理四元數符號，使其指向半球
                if (q.w() < 0) q.coeffs() *= -1;
                quats.push_back(q);
            }
            avg_t *= (1.0 / valid_poses.size());

            // 應用 Kalman 濾波到平均平移向量
            cv::Vec3d filtered_t = kalman_filter.update(avg_t);

            // 平均四元數 (Slerp 或簡單平均)
            // 簡單平均四元數 (如果所有四元數都在同一個半球)
            Quaterniond q_avg(0, 0, 0, 0);
            for (const auto& q : quats)
                q_avg.coeffs() += q.coeffs();
            q_avg.normalize(); // 歸一化平均後的四元數

            Matrix3d R_avg = q_avg.toRotationMatrix(); // 轉換為旋轉矩陣

            // 構建最終的平均姿態矩陣 (從世界坐標系到相機坐標系)
            cv::Mat T_avg = cv::Mat::eye(4, 4, CV_64F);
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    T_avg.at<double>(i, j) = R_avg(i, j);
            T_avg.at<double>(0, 3) = filtered_t[0];
            T_avg.at<double>(1, 3) = filtered_t[1];
            T_avg.at<double>(2, 3) = filtered_t[2];

            // 計算 Yaw 角
            // T_avg 是 (世界 -> 相機) 變換矩陣
            // 我們需要相機在世界坐標系中的姿態，即 (相機 -> 世界) = T_avg 的逆
            cv::Mat R_cam_to_world; // 相機坐標系到世界坐標系
            cv::transpose(T_avg(cv::Rect(0,0,3,3)), R_cam_to_world); // 旋轉矩陣的逆就是其轉置

            Eigen::Matrix3d R_cam_to_world_eigen;
            for(int i = 0; i < 3; ++i)
                for(int j = 0; j < 3; ++j)
                    R_cam_to_world_eigen(i,j) = R_cam_to_world.at<double>(i,j);

            // 提取 Euler 角 (Yaw, Pitch, Roll)
            // 這裡使用 ZYX 順序 (2,1,0)，即先繞 X (Roll)，再繞 Y (Pitch)，最後繞 Z (Yaw)
            // Yaw 軸通常是圍繞垂直於地平面的軸 (世界 Z 軸)
            // Eigen::eulerAngles(2,1,0) 返回 (Z, Y, X) 順序的弧度
            Eigen::Vector3d euler_angles = R_cam_to_world_eigen.eulerAngles(2, 1, 0);
            double yaw_rad = euler_angles[0]; // 提取 Yaw 角
            double yaw_deg = yaw_rad * 180.0 / CV_PI;

            cout << "\033[1;32m平均相機位置 (濾波後):\033[0m " << filtered_t << endl;
            cout << "平均相機姿態矩陣 (世界 -> 相機):\n" << T_avg << endl;
            cout << "平均相機 Yaw 角: " << yaw_deg << " 度" << endl;

            logFile << "濾波後相機位置: [" << filtered_t[0] << ", " << filtered_t[1] << ", " << filtered_t[2] << "]\n";
            logFile << "平均相機姿態矩陣 (世界 -> 相機):\n" << T_avg << endl;
            logFile << "平均相機 Yaw 角: " << yaw_deg << " 度\n";
            std::time_t now_c = std::time(nullptr);
            logFile << "現在時間: " << std::ctime(&now_c); // ctime 已經包含換行符

            // 繪製軸 (基於平均姿態)
            cv::Mat rvec_avg, tvec_avg;
            cv::Rodrigues(T_avg(cv::Rect(0,0,3,3)), rvec_avg);
            tvec_avg = T_avg(cv::Rect(3,0,1,3)).clone();
            cv::drawFrameAxes(frame, camParams.CameraMatrix, camParams.Distorsion, rvec_avg, tvec_avg, 0.1);


            // CURL 發送
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
        } else {
            // 如果沒有有效標記，可以選擇不清空 Kalamn 狀態或使用最後的已知狀態
            // 或者簡單地不更新位置
            std::cerr << "沒有偵測到有效標記用於姿態估計。" << std::endl;
        }
    }

    logFile.close();
    return 0;
}