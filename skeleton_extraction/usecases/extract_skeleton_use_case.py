from skeleton_extraction.domain.entities import SkeletonData
from skeleton_extraction.domain.extractor_interface import SkeletonExtractorInterface

class ExtractSkeletonUseCase:
    """
    Caso de uso para orquestar la extracción de skeletons de un video.
    """

    def __init__(self, extractor: SkeletonExtractorInterface):
        self.extractor = extractor

    def execute(self, video_path: str, config: dict = None) -> SkeletonData:
        return self.extractor.extract(video_path, config)
    
