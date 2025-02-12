from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class SkeletonData:
    """
    Representa los puntos clave extraídos de un video.
    - video_id: Identificador o ruta del video.
    - frames: Lista de diccionarios, donde cada diccionario representa
              los puntos clave de un frame (por ejemplo, 'frame_index' y 'keypoints').
    """
    video_id: str
    frames: List[Dict[str, Any]]
