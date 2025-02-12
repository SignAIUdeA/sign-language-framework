from typing import List
from data_loading.domain.entities import VideoData
from data_loading.domain.repositories import VideoRepositoryInterface

class LoadVideosUseCase:
    """
    Caso de uso para cargar la lista completa de videos.
    """

    def __init__(self, repository: VideoRepositoryInterface):
        self.repository = repository

    def execute(self) -> List[VideoData]:
        """
        Orquesta la carga de videos usando el repositorio inyectado.
        """
        return self.repository.get_all_videos()
