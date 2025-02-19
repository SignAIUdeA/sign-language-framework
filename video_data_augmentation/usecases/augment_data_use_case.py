from typing import List
from data_augmentation.domain.entities import VideoData, Transformation, AugmentedData
from data_augmentation.infrastructure.augmentation_service import AugmentationService

class AugmentDataUseCase:
    """
    Caso de uso para aplicar transformaciones de augmentación de datos a un video.
    """

    def __init__(self, augmentation_service: AugmentationService):
        """
        Inicializa el caso de uso con el servicio de augmentación.

        Args:
            augmentation_service (AugmentationService): El servicio de augmentación de datos.
        """
        self.augmentation_service = augmentation_service

    def execute(self, video: VideoData, transformations: List[Transformation]) -> AugmentedData:
        """
        Ejecuta el caso de uso para aplicar transformaciones a un video.

        Args:
            video (VideoData): El video original a ser transformado.
            transformations (List[Transformation]): Lista de transformaciones a aplicar.

        Returns:
            AugmentedData: Los datos del video aumentado.
        """
        return self.augmentation_service.augment_video(video, transformations)

    def get_available_transformations(self) -> List[Transformation]:
        """
        Obtiene la lista de transformaciones disponibles.

        Returns:
            List[Transformation]: Lista de transformaciones disponibles.
        """
        return self.augmentation_service.get_transformations()