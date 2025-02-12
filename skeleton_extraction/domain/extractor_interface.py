from abc import ABC, abstractmethod
from skeleton_extraction.domain.entities import SkeletonData

class SkeletonExtractorInterface(ABC):
    """
    Define la interfaz para extraer puntos clave (skeletons) de un video.
    """

    @abstractmethod
    def extract(self, video_path: str, config: dict = None) -> SkeletonData:
        """
        Procesa el video en 'video_path' y retorna los datos de skeleton.
        """
        pass
