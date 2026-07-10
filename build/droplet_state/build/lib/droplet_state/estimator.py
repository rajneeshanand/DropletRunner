#!/usr/bin/env python3


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np

from droplet_state.droplet_detector import DropletDetector
from droplet_state.marker_detector import MarkerDetector
from droplet_state.camera_model import OCamModel


class StateEstimator(Node):
    def __init__(self):
        super().__init__('state_estimator')

        self.bridge = CvBridge()
        self.detector = DropletDetector()
        self.marker_detector = MarkerDetector()

        #load OCamCalib camera model
        import os
        calib_path = os.path.join(
            os.path.dirname(__file__), '..', 'calib', 'calib_results.txt')
        self.cam_model = OCamModel(calib_path)
        self.get_logger().info(
            f'OCamCalib loaded: center=({self.cam_model.cx_scaled:.0f},'
            f'{self.cam_model.cy_scaled:.0f}), '
            f'scale=({self.cam_model.sx:.3f},{self.cam_model.sy:.3f})')

        #board ROI in pixels (tuned for our setup)
        self.roi_x1, self.roi_y1 = 338, 126
        self.roi_x2, self.roi_y2 = 908, 648

        #board dimensions (mm)
        self.board_width_mm = 266.0   # TL→TR
        self.board_height_mm = 241.0  # TL→BL

        #reference corner positions (undistorted) — computed on first frame
        self.ref_corners_undistorted = None
        self.homography = None

        #Kalman Filter for prediction during dropout
        # State: [x_mm, y_mm, vx_mm_s, vy_mm_s]
        # Measurement: [x_mm, y_mm]

        self.kf = cv2.KalmanFilter(4, 2)
        dt = 1.0 / 30.0                             # ~30 Hz camera rate

        #Transition matrix: x' = x + vx*dt, y' = y + vy*dt

        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float32)

        #we observe [x_mm, y_mm]
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        #process noise which allow for acceleration/direction changes
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 5.0
        self.kf.processNoiseCov[2, 2] = 50.0  # velocity can change fast
        self.kf.processNoiseCov[3, 3] = 50.0

        #measurement noise for detection jitter ~2mm
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4.0

        #initial covariance
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 100.0

        self.kf_initialized = False
        self.frames_since_detection = 0
        self.MAX_PREDICT_FRAMES = 90   # ~3 sec at 30 Hz

        self.image_sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10
        )

        self.state_pub = self.create_publisher(
            Float32MultiArray, '/droplet_state', 10
        )

        self.debug_pub = self.create_publisher(
            Image, '/debug_image', 10
        )

        self.get_logger().info(
            'State estimator ready (OCamCalib + Kalman prediction)')

    def _undistort_corners(self, corners_px):
  
        undistorted = []
        for (px, py) in corners_px:
            ux, uy = self.cam_model.undistort_point(float(px), float(py))
            undistorted.append((ux, uy))
        return undistorted

    def _compute_homography(self, corners_px):
  
        corners_undist = self._undistort_corners(corners_px)
        src = np.array(corners_undist, dtype=np.float64)

        W = self.board_width_mm
        H = self.board_height_mm
        dst = np.array([
            [-W/2, -H/2],  # TL
            [ W/2, -H/2],  # TR
            [ W/2,  H/2],  # BR
            [-W/2,  H/2],  # BL
        ], dtype=np.float64)

        H_mat, _ = cv2.findHomography(src, dst)
        return H_mat, corners_undist

    def _pixel_to_mm(self, px, py):

        if self.homography is None:
            return 0.0, 0.0

        ux, uy = self.cam_model.undistort_point(float(px), float(py))
        pt = np.array([ux, uy, 1.0])
        result = self.homography @ pt
        result /= result[2]

        return float(result[0]), float(result[1])

    def _estimate_tilt_from_corners(self, corners_px):

        if corners_px is None or len(corners_px) != 4:
            return 0.0, 0.0

        current_undist = self._undistort_corners(corners_px)

        if self.ref_corners_undistorted is None:
            self.ref_corners_undistorted = current_undist
            return 0.0, 0.0

        ref_center = np.mean(self.ref_corners_undistorted, axis=0)
        cur_center = np.mean(current_undist, axis=0)
        shift = np.array(cur_center) - np.array(ref_center)

        alpha = float(shift[0]) * 100.0
        beta = float(shift[1]) * 100.0

        return alpha, beta

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        #detect corner markers
        marker_margin = 30
        m_x1 = max(0, self.roi_x1 - marker_margin)
        m_y1 = max(0, self.roi_y1 - marker_margin)
        m_x2 = min(frame.shape[1], self.roi_x2 + marker_margin)
        m_y2 = min(frame.shape[0], self.roi_y2 + marker_margin)
        marker_region = frame[m_y1:m_y2, m_x1:m_x2]
        corners_local = self.marker_detector.detect(marker_region)

        if corners_local is not None:
            corners = [(x + m_x1, y + m_y1) for x, y in corners_local]
        else:
            corners = None

        if corners is not None and len(corners) == 4:
            if self.homography is None:
                self.homography, _ = self._compute_homography(corners)

        alpha, beta = self._estimate_tilt_from_corners(corners)

        #droplet in board region
        board_region = frame[self.roi_y1:self.roi_y2,
                             self.roi_x1:self.roi_x2]
        position = self.detector.detect(board_region)

        #Kalman filter update or predict
        state_msg = Float32MultiArray()

        if position is not None:
            cx = position[0] + self.roi_x1
            cy = position[1] + self.roi_y1
            x_mm, y_mm = self._pixel_to_mm(cx, cy)

            #update Kalman filter
            measurement = np.array(
                [[np.float32(x_mm)], [np.float32(y_mm)]])

            if not self.kf_initialized:
                #First detection: initialize state
                self.kf.statePost = np.array(
                    [[np.float32(x_mm)],
                     [np.float32(y_mm)],
                     [np.float32(0.0)],
                     [np.float32(0.0)]], dtype=np.float32)
                self.kf_initialized = True
            else:
                self.kf.predict()
                self.kf.correct(measurement)

            self.frames_since_detection = 0

            state_msg.data = [float(x_mm), float(y_mm),
                              float(alpha), float(beta), 1.0]

            self.get_logger().info(
                f'Droplet ({cx},{cy})px → ({x_mm:.1f},{y_mm:.1f})mm '
                f'tilt ({alpha:.1f},{beta:.1f})',
                throttle_duration_sec=1.0)

        elif self.kf_initialized and \
                self.frames_since_detection < self.MAX_PREDICT_FRAMES:
            #no detection, use Kalman prediction
            predicted = self.kf.predict()
            x_pred = float(predicted[0])
            y_pred = float(predicted[1])

            #Clamp to board bounds
            half_w = self.board_width_mm / 2.0
            half_h = self.board_height_mm / 2.0
            x_pred = max(-half_w, min(half_w, x_pred))
            y_pred = max(-half_h, min(half_h, y_pred))

            self.frames_since_detection += 1

            #detected = 0.5 signals "predicted, not measured"
            state_msg.data = [float(x_pred), float(y_pred),
                              float(alpha), float(beta), 0.5]

            self.get_logger().info(
                f'PREDICT ({x_pred:.1f},{y_pred:.1f})mm '
                f'[lost {self.frames_since_detection} frames] '
                f'tilt ({alpha:.1f},{beta:.1f})',
                throttle_duration_sec=1.0)

        else:
            #not initialized or prediction timed out
            state_msg.data = [0.0, 0.0,
                              float(alpha), float(beta), 0.0]

        self.state_pub.publish(state_msg)

        debug_frame = frame.copy()
        cv2.rectangle(debug_frame,
                      (self.roi_x1, self.roi_y1),
                      (self.roi_x2, self.roi_y2),
                      (255, 255, 0), 1)

        if corners is not None:
            for i, (mx, my) in enumerate(corners):
                cv2.circle(debug_frame, (int(mx), int(my)),
                           8, (255, 0, 0), 2)
                cv2.putText(debug_frame, str(i),
                           (int(mx) + 10, int(my)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        if position is not None:
            cx = position[0] + self.roi_x1
            cy = position[1] + self.roi_y1
            x_mm, y_mm = self._pixel_to_mm(cx, cy)
            cv2.circle(debug_frame, (int(cx), int(cy)),
                       10, (0, 255, 0), 2)
            cv2.putText(debug_frame,
                       f'({x_mm:.1f},{y_mm:.1f})mm',
                       (int(cx) + 15, int(cy)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        elif self.kf_initialized and \
                self.frames_since_detection < self.MAX_PREDICT_FRAMES:
            # Show predicted position in yellow
            kf_state = self.kf.statePost
            x_pred = float(kf_state[0])
            y_pred = float(kf_state[1])

            #convert predicted mm back to approx pixel for overlay
            half_w = self.board_width_mm / 2.0
            half_h = self.board_height_mm / 2.0
            px_approx = int(self.roi_x1 + (x_pred + half_w)
                            / self.board_width_mm
                            * (self.roi_x2 - self.roi_x1))
            py_approx = int(self.roi_y1 + (y_pred + half_h)
                            / self.board_height_mm
                            * (self.roi_y2 - self.roi_y1))

            cv2.circle(debug_frame, (px_approx, py_approx),
                       10, (0, 255, 255), 2)  # yellow = predicted
            cv2.putText(debug_frame,
                       f'PRED ({x_pred:.1f},{y_pred:.1f})mm',
                       (px_approx + 15, py_approx),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 255, 255), 1)

        status = 'DETECTED' if position is not None else (
            f'PREDICT [{self.frames_since_detection}]'
            if self.kf_initialized
            and self.frames_since_detection < self.MAX_PREDICT_FRAMES
            else 'LOST')

        cv2.putText(debug_frame,
                   f'a={alpha:.1f} b={beta:.1f} [{status}]',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 255), 2)

        debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, 'bgr8')
        self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = StateEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()