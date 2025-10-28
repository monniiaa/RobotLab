# explore_landmarks.py
from Utils.CalibratedRobot import CalibratedRobot
import time
import numpy as np

import time

class LocalizationPathing:
    def __init__(self, robot, required_landmarks, step_cm=20, rotation_deg=20, min_landmarks_to_see = 2):
        self.robot = robot
        self.required_landmarks = set(required_landmarks)
        self.step_cm = step_cm
        self.rotation_deg = rotation_deg
        self.min_landmarks_to_see = min_landmarks_to_see

        self.observed_landmarks = set()
        self.min_landmarks_met = False

    def explore_step(self, drive=False, min_dist = 400):
        dist = 0
        angle_deg = self.rotation_deg 
        angle_rad = np.radians(angle_deg)

        if not drive:
            self.robot.turn_angle(angle_deg)
            angle_rad = np.radians(self.rotation_deg)
            time.sleep(0.2)

        if drive:
            dist = self.step_cm
            left, center, right = self.robot.proximity_check()

            if left < min_dist or center < min_dist or right < min_dist:
                self.robot.stop()
            if left > right:
                self.robot.turn_angle(45)   
                angle_rad = np.radians(45)
            else:
                self.robot.turn_angle(-45)
                angle_rad = np.radians(-45)

            dist, obstacle_detected = self.robot.drive_distance_cm(dist)

        return dist, angle_rad, obstacle_detected
    
    def saw_landmark(self, landmarkID):
        """
        Register that a landmark with a given ID has been seen.
        """
        if landmarkID in self.required_landmarks:
            self.observed_landmarks.add(landmarkID)

            if len(self.observed_landmarks) >= self.min_landmarks_to_see:
                self.min_landmarks_met = True

    def seen_enough_landmarks(self):
        """
        Returns True if at least `min_landmarks_seen` have been observed.
        """
        return self.min_landmarks_met

    
    def move_towards_goal_step(self, est_pose, goal):
        robot_pos = np.array([est_pose.getX(), est_pose.getY()])
        direction = goal - robot_pos
        distance_to_goal = np.linalg.norm(direction)
        angle_to_goal = np.arctan2(direction[1], direction[0]) - est_pose.getTheta()
        
        move_dist = distance_to_goal - 12.5
        angle_to_goal = (angle_to_goal + np.pi) % (2 * np.pi) - np.pi
        
        print(f"distance moved: {distance_to_goal}")
        print(f"angle (rad) turned: {angle_to_goal}")

        self.robot.turn_angle(np.degrees(angle_to_goal))

        distance, obstacleDetected = self.robot.drive_distance_cm(move_dist)
        self.robot.stop()
        time.sleep(0.2)

        return distance, angle_to_goal, obstacleDetected


def avoid_obstacle(self,  dist = 10, turn_angle=45, stop_threshold=25):
    """
    Reads proximity sensors and performs a short avoidance maneuver toward
    the side with more free space.
    """
    left, center, right = self.proximity_check()
    angle = 0
    distance = 0
    # Determine which side is more open
    if left > right:
        self.turn_angle(turn_angle)
        angle = turn_angle

    else:
        self.turn_angle(-turn_angle)
        angle = -turn_angle

    # Optionally, check again after turning
    left, center, right = self.proximity_check()

    # If still too close, back up a bit
    if min(left, center, right) < stop_threshold:
        print("Still too close, backing up")
        self.drive_distance_cm(dist, direction=self.BACKWARD)
        distance = dist

    return distance, np.radians(angle)
