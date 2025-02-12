from dataclasses import dataclass
from typing import Optional

@dataclass
class VideoData:
    """
    Entidad que representa la información básica de un video.
    """
    video_id: str
    file_path: str
    label: Optional[str] = None
    additional_info: dict = None
