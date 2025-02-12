import os
import csv
import uuid
import json
from typing import List
from data_loading.domain.entities import VideoData
from data_loading.domain.repositories import VideoRepositoryInterface

SUPPORTED_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

class FileSystemVideoRepository(VideoRepositoryInterface):
    """
    Implementación del repositorio que carga videos a partir de una carpeta raíz.
    La estructura asumida es:
        data/
          ├─ S1/
          │    ├─ video1.mp4
          │    └─ video2.mp4
          └─ S2/
               ├─ video1.mp4
               └─ video2.mp4
    Si se proporciona un archivo de cache para metadata, se leerá desde él si existe; 
    de lo contrario se generará la metadata y se almacenará en dicho archivo.
    """

    def __init__(self, data_folder: str, metadata_cache_file: str = None):
        """
        :param data_folder: Ruta raíz donde se encuentran las carpetas de clases.
        :param metadata_cache_file: Ruta del archivo CSV para almacenar la metadata (opcional).
        """
        self.data_folder = data_folder
        self.metadata_cache_file = metadata_cache_file

        if metadata_cache_file and os.path.exists(metadata_cache_file):
            self._videos_data_cache = self._load_metadata_from_cache(metadata_cache_file)
        else:
            self._videos_data_cache = self._scan_data_folder()
            if metadata_cache_file:
                self._save_metadata_to_cache(metadata_cache_file, self._videos_data_cache)

    def _scan_data_folder(self) -> List[VideoData]:
        """
        Escanea la carpeta raíz, interpretando cada subcarpeta como una clase,
        y genera la metadata de cada video encontrado.
        """
        videos_data = []

        for class_name in os.listdir(self.data_folder):
            class_path = os.path.join(self.data_folder, class_name)
            if os.path.isdir(class_path):
                for file_name in os.listdir(class_path):
                    if file_name.lower().endswith(SUPPORTED_EXTENSIONS):
                        file_path = os.path.join(class_path, file_name)

                        video_id = str(uuid.uuid4())

                        try:
                            file_size = os.path.getsize(file_path)
                        except Exception:
                            file_size = None
                        additional_info = {"file_size": file_size}
                        video_data = VideoData(
                            video_id=video_id,
                            file_path=file_path,
                            label=class_name,
                            additional_info=additional_info
                        )
                        videos_data.append(video_data)
        return videos_data

    def _save_metadata_to_cache(self, metadata_cache_file: str, videos_data: List[VideoData]):
        """
        Guarda la metadata generada en un archivo CSV para optimizar futuros accesos.
        """
        with open(metadata_cache_file, mode='w', encoding='utf-8', newline='') as csv_file:
            fieldnames = ["video_id", "file_path", "label", "additional_info"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for video in videos_data:
                writer.writerow({
                    "video_id": video.video_id,
                    "file_path": video.file_path,
                    "label": video.label,
                    "additional_info": json.dumps(video.additional_info)
                })

    def _load_metadata_from_cache(self, metadata_cache_file: str) -> List[VideoData]:
        """
        Carga la metadata previamente guardada desde un archivo CSV.
        """
        videos_data = []
        with open(metadata_cache_file, mode='r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                video_data = VideoData(
                    video_id=row["video_id"],
                    file_path=row["file_path"],
                    label=row["label"],
                    additional_info=json.loads(row["additional_info"]) if row.get("additional_info") else {}
                )
                videos_data.append(video_data)
        return videos_data

    def get_all_videos(self) -> List[VideoData]:
        return self._videos_data_cache

    def get_video_by_id(self, video_id: str) -> VideoData:
        for video_data in self._videos_data_cache:
            if video_data.video_id == video_id:
                return video_data
        raise ValueError(f"No se encontró el video con ID: {video_id}")
