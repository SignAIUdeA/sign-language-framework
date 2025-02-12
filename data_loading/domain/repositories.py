from abc import ABC, abstractmethod
from typing import List
from .entities import VideoData

class VideoRepositoryInterface(ABC):
    """
    Define la interfaz para cargar y manipular datos de videos.
    """

    @abstractmethod
    def get_all_videos(self) -> List[VideoData]:
        """
        Retorna la lista de todos los videos disponibles.
        """
        pass

    @abstractmethod
    def get_video_by_id(self, video_id: str) -> VideoData:
        """
        Retorna un video específico por su ID.
        """
        pass
