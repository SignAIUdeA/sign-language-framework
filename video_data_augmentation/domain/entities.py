from dataclasses import dataclass
from typing import Optional, List

@dataclass
class VideoData:
    """
    Entidad que representa la información básica de un video.
    """
    video_id: str
    file_path: str
    label: Optional[str] = None
    additional_info: dict = None

@dataclass
class AugmentedData:
    """
    Entidad que representa los datos aumentados.
    """
    original_video_id: str
    augmented_video_id: str
    transformations: List[str]
    file_path: str
    label: Optional[str] = None
    additional_info: dict = None

@dataclass
class Transformation:
    """
    Entidad que representa una transformación aplicada a los datos originales.
    """
    name: str
    parameters: dict