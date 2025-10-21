import time
import numpy as np
from RobotUtils.CameraUtils import CameraUtils, ArucoUtils
import cv2
from RobotUtils.CalibratedRobot import CalibratedRobot
import matplotlib.pyplot as plt

class FindLandmarks:
    def __init__(self):
        self.calArlo = CalibratedRobot()
        self.cam = CameraUtils()
        self.aruco = ArucoUtils()
        self.cam.start_camera()
        self.isDriving = False
        self.last_id = None

    def stop(self):
        """Stop robot and camera safely."""
        self.calArlo.stop()
        self.cam.stop_camera()

    def drive_to_landmark(self):
        """Drive to the nearest detected Aruco marker."""
        self.isDriving = False
        self.last_id = None

        while True:
            frame = self.cam.get_frame()
            corners, ids = self.aruco.detect_markers(frame)
            if ids is not None:
                marker_id = int(ids[0][0])
                print(f"id found: {marker_id}")
                rvecs, tvecs = self.aruco.estimate_pose(corners, self.cam.camera_matrix)
                tvec = tvecs[0][0]

                dist = self.aruco.compute_distance_to_marker(tvec)
                angle = self.aruco.compute_rotation_to_marker(tvec)
                print(f"distance: {dist}")
                print(f"angle: {angle}")

                self.calArlo.turn_angle(angle)

                if not self.isDriving and marker_id != self.last_id:
                    self.isDriving = True
                    self.calArlo.drive_distance(dist)

                if dist <= 0:
                    self.last_id = marker_id
                    self.isDriving = False
            else:
                self.calArlo.turn_angle(15)
                time.sleep(0.2)

    def drive_to_landmark_steps(self, landmark, step_distance=40):
        """Drive to a marker in smaller steps for more control."""
        moves = []
        while True:
            frame = self.cam.get_frame()
            corners, ids = self.aruco.detect_markers(frame)

            if ids is not None and landmark in ids:
            # Get pose of the specific landmark
                index = list(ids).index(landmark)
                rvecs, tvecs = self.aruco.estimate_pose(corners, self.cam.camera_matrix)
                tvec = tvecs[index][0]

                dist = self.aruco.compute_distance_to_marker(tvec)
                angle = self.aruco.compute_rotation_to_marker(tvec)
                print(f"distance: {dist}")
                print(f"angle: {angle}")

                self.calArlo.turn_angle(angle)
                moves.append((0, np.radians(angle))) 
                while dist > 30:
                    drive_step = min(step_distance, dist)
                    print(f"Drove distance: {drive_step}")
                    self.calArlo.drive_distance_cm(drive_step)
                    moves.append((drive_step, 0))
                    time.sleep(0.05)

                    frame = self.cam.get_frame()
                    corners, ids = self.aruco.detect_markers(frame)
                    if ids is None:
                        print("Lost marker")
                        break
                    tvec = self.aruco.estimate_pose(corners, self.cam.camera_matrix)[1][0][0]
                    dist = self.aruco.compute_distance_to_marker(tvec)
                    angle = self.aruco.compute_rotation_to_marker(tvec)
                    self.calArlo.turn_angle(angle)

                    self.calArlo.turn_angle(angle)
                    moves.append((0, np.radians(angle)))

                    if dist <= 30:
                        print("Reached landmark!")
                        return moves
            else:
                self.calArlo.turn_angle(20)
                moves.append((0, np.radians(angle)))
                time.sleep(0.2)

