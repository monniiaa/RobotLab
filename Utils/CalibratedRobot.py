import time
import numpy as np
import math
import Utils.Robot as robot
from Utils.vector_utils import VectorUtils

class CalibratedRobot:
    def __init__(self):
        self.arlo = robot.Robot()

        self.TRANSLATION_TIME = 2.5 # time to drive 1 meter at default speed (64)
        self.TURN_TIME = 0.80 # time to turn 90 degrees at default speed (64)
        
        #ratio for adjusting the wheels to have the same power
        self.CAL_KL = 0.99
        self.CAL_KR = 1.0
        
        self.MIN_PWR = 40
        self.MAX_PWR = 127

        self.default_speed = 64  
        self.FORWARD = 1
        self.BACKWARD = 0
        
        self.stop_distance = 200

    def clamp_power(self, p):
        return max(self.MIN_PWR, min(self.MAX_PWR, int(round(p))))
    
    def drive(self, leftSpeed, rightSpeed, leftDir, rightDir):
        l = self.clamp_power(leftSpeed * self.CAL_KL) if leftSpeed > 0 else 0
        r = self.clamp_power(rightSpeed * self.CAL_KR) if rightSpeed > 0 else 0
        self.arlo.go_diff(l, r, leftDir, rightDir)

    def drive_distance(self, meters, direction=None, speed=None, stop_threshold=25):
        """Drive a certain distance in meters with proximity safety."""
        if speed is None:
            speed = self.default_speed
        if direction is None:
            direction = self.FORWARD

        duration = self.TRANSLATION_TIME * meters * (self.default_speed / speed)
        start_time = time.time()
        obstacle_detected = False

        self.drive(speed, speed, direction, direction)

        while True:
            left, center, right = self.proximity_check()
            if min(left, center, right) < stop_threshold:
                print("Obstacle detected, stopping.")
                obstacle_detected = True
                break

            if time.time() - start_time >= duration:
                break

            time.sleep(0.05)

        self.arlo.stop()

        # compute how far we actually drove
        elapsed = time.time() - start_time
        actual_meters = (elapsed / duration) * meters

        return actual_meters, obstacle_detected



    def drive_distance_cm(self, distance_cm, direction=None, speed=None):
        distance_m = distance_cm / 100.0
        actual_meters, obstacleDetected = self.drive_distance(distance_m, direction, speed)
        actual_cm = actual_meters * 100.0
        return actual_cm, obstacleDetected


    def turn_angle(self, angleDeg, speed=None):
        """Turn a given angle in degrees at a given speed. Positive = left, negative = right."""
        if speed is None:
            speed = self.default_speed
        #The formula for the duration to turn the desired angle: duration = TURN_TIME *  (abs(angle) / 90.0) * (default_speed /current speed)
        duration = self.TURN_TIME * (abs(angleDeg) / 90.0) * (self.default_speed / speed)
        if angleDeg > 0:
            self.drive(speed, speed, self.BACKWARD, self.FORWARD)  # left
            time.sleep(duration)
            self.arlo.stop()
        else:
            self.drive(speed, speed, self.FORWARD, self.BACKWARD)  # right
            time.sleep(duration)
            self.arlo.stop()
            
    def proximity_check(self):
        left = self.arlo.read_left_ping_sensor()
        center = self.arlo.read_front_ping_sensor()
        right = self.arlo.read_right_ping_sensor()
        
        return  left, center, right
    
    def follow_path(self, path, start_orientation=np.array([0, 1])):
        moves = []
        orientation_unit = start_orientation / np.linalg.norm(start_orientation)
        obstacleDetected = False

        for i in range(len(path) - 1):
            current_p = np.array(path[i])
            next_p = np.array(path[i + 1])

            next_vec = next_p - current_p
            next_unit = next_vec / np.linalg.norm(next_vec)

            dot = np.clip(np.dot(orientation_unit, next_unit), -1.0, 1.0)
            angle = np.arccos(dot)

            cross = orientation_unit[0]*next_unit[1] - orientation_unit[1]*next_unit[0]
            if cross < 0:
                angle = -angle 

            orientation_unit = VectorUtils.rotate_vector(orientation_unit, angle)
            orientation_unit /= np.linalg.norm(orientation_unit)

            distance = np.linalg.norm(next_vec)

            print(f"angle: {math.degrees(angle):.2f} degrees")
            print(f"distance: {distance:.2f}")

            self.turn_angle(math.degrees(angle))
            distance, obstacleDetected = self.drive_distance_cm(distance)

            if obstacleDetected: 
                break

            moves.append((distance, angle))
            
        return moves, obstacleDetected
            
    def stop(self):
        self.arlo.stop()