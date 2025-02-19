from typing import List
from video_data_augmentation.domain.entities import VideoData, AugmentedData, Transformation
from video_data_augmentation.domain.repositories import DataAugmentationRepositoryInterface
from video_data_augmentation.infrastructure.transformations import (
    apply_piecewise_affine_transform,
    apply_superpixel,
    apply_gaussian_blur,
    apply_invert_color,
    apply_random_rotate,
    apply_random_resize,
    apply_translate,
    apply_center_crop,
    apply_horizontal_flip,
    apply_vertical_flip,
    apply_add,
    apply_multiply,
    apply_downsample,
    apply_upsample,
    apply_elastic_transformation,
    apply_salt,
    apply_pepper,
    apply_shear,
    apply_mirror
)
class AugmentationService(DataAugmentationRepositoryInterface):
    def augment_video(self, video: VideoData, transformations: List[Transformation]) -> AugmentedData:
        """
        Implementa la lógica para aplicar transformaciones a un video.
        """
        augmented_video_id = f"{video.video_id}_augmented"
        file_path = f"{video.file_path}_augmented"
        
        # Aplicar cada transformación seleccionada al video
        for transformation in transformations:
            if transformation.name == "mirror":
                video = apply_mirror(video, transformation.parameters)
            elif transformation.name == "piecewise_affine_transform":
                video = apply_piecewise_affine_transform(video, transformation.parameters)
            elif transformation.name == "superpixel":
                video = apply_superpixel(video, transformation.parameters)
            elif transformation.name == "gaussian_blur":
                video = apply_gaussian_blur(video, transformation.parameters)
            elif transformation.name == "invert_color":
                video = apply_invert_color(video, transformation.parameters)
            elif transformation.name == "random_rotate":
                video = apply_random_rotate(video, transformation.parameters)
            elif transformation.name == "random_resize":
                video = apply_random_resize(video, transformation.parameters)
            elif transformation.name == "translate":
                video = apply_translate(video, transformation.parameters)
            elif transformation.name == "center_crop":
                video = apply_center_crop(video, transformation.parameters)
            elif transformation.name == "horizontal_flip":
                video = apply_horizontal_flip(video, transformation.parameters)
            elif transformation.name == "vertical_flip":
                video = apply_vertical_flip(video, transformation.parameters)
            elif transformation.name == "add":
                video = apply_add(video, transformation.parameters)
            elif transformation.name == "multiply":
                video = apply_multiply(video, transformation.parameters)
            elif transformation.name == "downsample":
                video = apply_downsample(video, transformation.parameters)
            elif transformation.name == "upsample":
                video = apply_upsample(video, transformation.parameters)
            elif transformation.name == "elastic_transformation":
                video = apply_elastic_transformation(video, transformation.parameters)
            elif transformation.name == "salt":
                video = apply_salt(video, transformation.parameters)
            elif transformation.name == "pepper":
                video = apply_pepper(video, transformation.parameters)
            elif transformation.name == "shear":
                video = apply_shear(video, transformation.parameters)

        augmented_data = AugmentedData(
            original_video_id=video.video_id,
            augmented_video_id=augmented_video_id,
            transformations=[t.name for t in transformations],
            file_path=file_path,
            label=video.label,
            additional_info=video.additional_info
        )
        return augmented_data

    def get_transformations(self) -> List[Transformation]:
        """
        Devuelve una lista de transformaciones disponibles.
        """
        return [
            Transformation(name="mirror", parameters={}),
            Transformation(name="piecewise_affine_transform", parameters={}),
            Transformation(name="superpixel", parameters={}),
            Transformation(name="gaussian_blur", parameters={}),
            Transformation(name="invert_color", parameters={}),
            Transformation(name="random_rotate", parameters={"angle": 15}),
            Transformation(name="random_resize", parameters={"scale": 1.1}),
            Transformation(name="translate", parameters={"x": 10, "y": 10}),
            Transformation(name="center_crop", parameters={"width": 100, "height": 100}),
            Transformation(name="horizontal_flip", parameters={}),
            Transformation(name="vertical_flip", parameters={}),
            Transformation(name="add", parameters={"value": 10}),
            Transformation(name="multiply", parameters={"value": 1.1}),
            Transformation(name="downsample", parameters={"factor": 2}),
            Transformation(name="upsample", parameters={"factor": 2}),
            Transformation(name="elastic_transformation", parameters={"alpha": 1.0, "sigma": 0.5}),
            Transformation(name="salt", parameters={"amount": 0.05}),
            Transformation(name="pepper", parameters={"amount": 0.05}),
            Transformation(name="shear", parameters={"angle": 10}),
        ]