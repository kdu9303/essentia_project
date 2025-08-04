# https://essentia.upf.edu/models.html#feature-extractors
import os
import numpy as np
from typing import Dict, Any, Optional, Tuple
from essentia.standard import (
    MonoLoader,  # type: ignore
    TensorflowPredictVGGish,  # type: ignore
    TensorflowPredictMusiCNN,  # type: ignore
    TensorflowPredictEffnetDiscogs,  # type: ignore
)


class FeatureExtractor:
    """
    Base class for extracting embeddings from audio files.
    """

    def __init__(self, audio_file: str):
        """
        :param audio_file: Path to the audio file.
        """
        self.audio_file = audio_file
        self.sample_rate = 16000
        self.resample_quality = 4
        self.audio: Optional[np.ndarray] = None
        self._load_audio()

    def _load_audio(self) -> None:
        """
        오디오 파일 로드
        :raises FileNotFoundError: 파일이 존재하지 않는 경우
        :raises ValueError: 오디오 로드에 실패한 경우
        """
        if not os.path.exists(self.audio_file):
            raise FileNotFoundError(
                f"오디오 파일을 찾을 수 없습니다: {self.audio_file}"
            )

        if not os.path.isfile(self.audio_file):
            raise ValueError(f"유효한 파일이 아닙니다: {self.audio_file}")

        try:
            loader = MonoLoader(
                filename=self.audio_file,
                sampleRate=self.sample_rate,
                resampleQuality=self.resample_quality,
            )
            self.audio = loader()

            if self.audio is None or len(self.audio) == 0:
                raise ValueError(
                    f"오디오 데이터를 로드할 수 없습니다: {self.audio_file}"
                )

        except Exception as e:
            raise ValueError(
                f"오디오 파일 로드 중 오류 발생: {self.audio_file}, 오류: {str(e)}"
            )

    def get_vggish_embeddings(self) -> Any:
        """AudioSet-VGGish"""
        if self.audio is None:
            raise ValueError("오디오 데이터가 로드되지 않았습니다.")

        try:
            model = TensorflowPredictVGGish(
                graphFilename="embeddings/audioset-vggish-3.pb",
                output="model/vggish/embeddings",
            )
            return model(self.audio)
        except Exception as e:
            raise RuntimeError(f"VGGish embedding 추출 중 오류 발생: {str(e)}")

    def get_yamnet_embeddings(self) -> Any:
        """AudioSet-VGGish"""
        if self.audio is None:
            raise ValueError("오디오 데이터가 로드되지 않았습니다.")

        try:
            model = TensorflowPredictVGGish(
                graphFilename="embeddings/audioset-yamnet-1.pb",
                input="melspectrogram",
                output="embeddings",
            )
            return model(self.audio)
        except Exception as e:
            raise RuntimeError(f"VGGish embedding 추출 중 오류 발생: {str(e)}")

    def get_effnet_embeddings(
        self, model_file: str = "embeddings/discogs-effnet-bs64-1.pb"
    ) -> Any:
        """
        effnet-discogs

        discogs_label_embeddings-effnet-bs64
        discogs_multi_embeddings-effnet-bs64
        discogs_release_embeddings-effnet-bs64
        discogs_track_embeddings-effnet-bs64
        """
        if self.audio is None:
            raise ValueError("오디오 데이터가 로드되지 않았습니다.")

        try:
            model = TensorflowPredictEffnetDiscogs(
                graphFilename=model_file, output="PartitionedCall:1"
            )
            return model(self.audio)
        except Exception as e:
            raise RuntimeError(f"Effnet embedding 추출 중 오류 발생: {str(e)}")

    def get_musicnn_embeddings(self) -> Any:
        """MusiCNN"""
        if self.audio is None:
            raise ValueError("오디오 데이터가 로드되지 않았습니다.")

        try:
            model = TensorflowPredictMusiCNN(
                graphFilename="embeddings/msd-musicnn-1.pb", output="model/dense/BiasAdd"
            )
            return model(self.audio)
        except Exception as e:
            raise RuntimeError(f"MusiCNN embedding 추출 중 오류 발생: {str(e)}")
