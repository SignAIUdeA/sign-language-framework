import cv2
import mediapipe as mp
from skeleton_extraction.domain.entities import SkeletonData
from skeleton_extraction.domain.extractor_interface import SkeletonExtractorInterface

class MediaPipeSkeletonExtractor(SkeletonExtractorInterface):
    """
    Implementa la extracción de puntos clave usando MediaPipe.
    Permite configurar distintos parámetros (por ejemplo, model_complexity).
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.config.get("model_complexity", 1),
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=self.config.get("min_detection_confidence", 0.5),
            min_tracking_confidence=self.config.get("min_tracking_confidence", 0.5)
        )

    def extract(self, video_path: str, config: dict = None) -> SkeletonData:
        cap = cv2.VideoCapture(video_path)
        frames_data = []
        frame_idx = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break


            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            keypoints = {}

            if results.pose_landmarks:

                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    keypoints[f"landmark_{idx}"] = (landmark.x, landmark.y, landmark.z)
            
            frames_data.append({
                "frame_index": frame_idx,
                "keypoints": keypoints
            })
            frame_idx += 1

        cap.release()
        return SkeletonData(video_id=video_path, frames=frames_data)
