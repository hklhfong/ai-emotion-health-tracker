import cv2
import mediapipe as mp
import numpy as np

class FaceEngine:
    """
    Research-Grade Face & Eye Gaze Tracking Engine:
    - 468 3D facial landmarks + 10 refined Iris Landmarks (468-477)
    - 3D Head Pose estimation (Pitch, Yaw, Roll via cv2.solvePnP)
    - Iris Gaze Tracking: Measures relative pupil position inside the eye socket
      (Horizontal & Vertical eye gaze vector)
    - Dual Head-Pose + Eye Gaze Fusion for robust Screen Focus / Attention scoring.
    """
    def __init__(self, max_num_faces=1):
        self.mp_face_mesh = mp.solutions.face_mesh
        # refine_landmarks=True enables Iris landmark tracking (MediaPipe 468-477 points)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 3D Model generic reference points for head pose
        self.model_points_3d = np.array([
            (0.0, 0.0, 0.0),             # Nose tip (1)
            (0.0, -330.0, -65.0),        # Chin (152)
            (-225.0, 170.0, -135.0),     # Left eye left corner (33)
            (225.0, 170.0, -135.0),      # Right eye right corner (263)
            (-150.0, -150.0, -125.0),    # Left Mouth corner (61)
            (150.0, -150.0, -125.0)      # Right mouth corner (291)
        ], dtype=np.float64)
        
        self.pose_indices = [1, 152, 33, 263, 61, 291]
        
        # Iris & Eye corner landmark indices
        # Left Eye (from user perspective)
        self.LEFT_IRIS_CENTER = 468
        self.LEFT_EYE_INNER = 362
        self.LEFT_EYE_OUTER = 263
        self.LEFT_EYE_TOP = 386
        self.LEFT_EYE_BOTTOM = 374
        
        # Right Eye (from user perspective)
        self.RIGHT_IRIS_CENTER = 473
        self.RIGHT_EYE_INNER = 133
        self.RIGHT_EYE_OUTER = 33
        self.RIGHT_EYE_TOP = 159
        self.RIGHT_EYE_BOTTOM = 145

    def process_frame(self, frame_bgr):
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.face_mesh.process(rgb_frame)

    def estimate_eye_gaze(self, landmarks, w, h):
        """
        Calculates Eye Gaze Vector (horizontal & vertical gaze ratios: 0.0 - 1.0, 0.5 = center looking at screen).
        Returns:
            gaze_x_ratio: < 0.4 = looking left, > 0.6 = looking right, ~0.5 = looking at screen
            gaze_y_ratio: < 0.4 = looking up, > 0.6 = looking down
            gaze_deviation: 0.0 (direct screen focus) to 1.0 (looking far off-screen)
        """
        try:
            if len(landmarks) < 478:
                return 0.5, 0.5, 0.0

            # Left eye iris position
            l_iris = landmarks[self.LEFT_IRIS_CENTER]
            l_inner = landmarks[self.LEFT_EYE_INNER]
            l_outer = landmarks[self.LEFT_EYE_OUTER]
            l_top = landmarks[self.LEFT_EYE_TOP]
            l_bot = landmarks[self.LEFT_EYE_BOTTOM]

            # Right eye iris position
            r_iris = landmarks[self.RIGHT_IRIS_CENTER]
            r_inner = landmarks[self.RIGHT_EYE_INNER]
            r_outer = landmarks[self.RIGHT_EYE_OUTER]
            r_top = landmarks[self.RIGHT_EYE_TOP]
            r_bot = landmarks[self.RIGHT_EYE_BOTTOM]

            # Compute horizontal gaze ratios
            # (distance iris to outer corner) / (total eye width)
            l_width = np.linalg.norm(np.array([l_inner.x, l_inner.y]) - np.array([l_outer.x, l_outer.y]))
            r_width = np.linalg.norm(np.array([r_inner.x, r_inner.y]) - np.array([r_outer.x, r_outer.y]))
            
            if l_width == 0 or r_width == 0:
                return 0.5, 0.5, 0.0

            l_gaze_h = (l_iris.x - min(l_inner.x, l_outer.x)) / l_width
            r_gaze_h = (r_iris.x - min(r_inner.x, r_outer.x)) / r_width
            avg_gaze_x = (l_gaze_h + r_gaze_h) / 2.0

            # Compute vertical gaze ratio
            l_height = np.linalg.norm(np.array([l_top.x, l_top.y]) - np.array([l_bot.x, l_bot.y]))
            r_height = np.linalg.norm(np.array([r_top.x, r_top.y]) - np.array([r_bot.x, r_bot.y]))
            
            if l_height == 0 or r_height == 0:
                avg_gaze_y = 0.5
            else:
                l_gaze_v = (l_iris.y - min(l_top.y, l_bot.y)) / l_height
                r_gaze_v = (r_iris.y - min(r_top.y, r_bot.y)) / r_height
                avg_gaze_y = (l_gaze_v + r_gaze_v) / 2.0

            # Gaze deviation from screen center (0.5, 0.5)
            # Normal desk screen viewing ranges between 0.38 and 0.62
            dev_x = max(0.0, abs(avg_gaze_x - 0.5) - 0.12)
            dev_y = max(0.0, abs(avg_gaze_y - 0.5) - 0.12)
            gaze_deviation = np.sqrt(dev_x**2 + dev_y**2)

            return float(avg_gaze_x), float(avg_gaze_y), float(gaze_deviation)
        except Exception:
            return 0.5, 0.5, 0.0

    def estimate_head_pose_and_focus(self, landmarks, w, h):
        """
        Fuses 3D Head Pose + Iris Gaze Tracking to calculate the overall Focus Score.
        Even if your head is slightly turned, if your eyes are looking at the screen, focus stays high.
        If your head is facing forward but eyes are looking away/at phone, focus drops accurately.
        """
        image_points = np.array([
            [landmarks[idx].x * w, landmarks[idx].y * h] for idx in self.pose_indices
        ], dtype=np.float64)

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rot_vec, trans_vec = cv2.solvePnP(
            self.model_points_3d, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return 0.0, 0.0, 0.0, 100.0, 0.5, 0.5

        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        raw_pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])
        
        if raw_pitch > 90.0:
            pitch = raw_pitch - 180.0
        elif raw_pitch < -90.0:
            pitch = raw_pitch + 180.0
        else:
            pitch = raw_pitch

        # 2. Eye Gaze Tracking
        gaze_x, gaze_y, gaze_dev = self.estimate_eye_gaze(landmarks, w, h)

        # 3. Head Pose Deviation
        dev_pitch = max(0.0, abs(pitch) - 7.0)
        dev_yaw = max(0.0, abs(yaw) - 9.0)
        dev_roll = max(0.0, abs(roll) - 6.0)
        head_dev = np.sqrt((dev_pitch * 1.3)**2 + (dev_yaw * 1.8)**2 + (dev_roll * 0.8)**2)

        # 4. Multimodal Attention Fusion (60% Head Pose + 40% Eye Gaze Vector)
        # Eye gaze deviation scaled by 150
        total_deviation = (0.60 * head_dev) + (0.40 * gaze_dev * 150.0)
        focus_score = max(0.0, min(100.0, 100.0 - total_deviation * 2.2))
        
        return pitch, yaw, roll, focus_score, gaze_x, gaze_y
