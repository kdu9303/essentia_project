# https://essentia.upf.edu/models.html#classifiers
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from essentia.standard import (
    MonoLoader,  # type: ignore
    TensorflowPredictVGGish,  # type: ignore
    TensorflowPredictMusiCNN,  # type: ignore
    TensorflowPredictEffnetDiscogs,  # type: ignore
    TensorflowPredict2D,  # type: ignore
)
from feature_extractors import FeatureExtractor


class AudioClassifier(FeatureExtractor):
    """transfer learning classifiers"""

    def __init__(self, audio_file: str):
        self.audio_file = audio_file
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

        super().__init__(self.audio_file)

    def _dowonload_models(self, url) -> None:
        """
        :param url: 모델 파일의 URL
        """
        import urllib.request

        # URL에서 파일명 추출
        filename = url.split("/")[-1]
        filepath = self.models_dir / filename

        # 파일이 이미 존재하는지 확인
        if filepath.exists():
            # print(f"모델 파일이 이미 존재합니다: {filepath}")
            return

        try:
            print(f"모델 다운로드 중: {filename}")
            urllib.request.urlretrieve(url, filepath)
            print(f"다운로드 완료: {filepath}")
        except Exception as e:
            print(f"다운로드 실패: {e}")
            raise

    def _select_embeddings(self, model_name: str) -> Any:
        """
        모델명을 기반으로 임베딩 선택
        :param model_name: 임베딩 알고리즘 이름
        :return: 선택된 임베딩
        """
        if "audioset-vggish-3" in model_name:
            return self.get_vggish_embeddings()
        elif "msd-musicnn-1" in model_name:
            return self.get_musicnn_embeddings()
        elif "discogs-effnet-bs64-1" in model_name:
            return self.get_effnet_embeddings()
        elif "audioset-yamnet-1" in model_name:
            return self.get_yamnet_embeddings()
        else:
            raise ValueError(f"지원되지 않는 모델명: {model_name}")

    def _select_classifier(
        self,
        classifier_model: str,
        classifier_name: str,
        classifier_output: str,
        classifier_input: Optional[str] = None,
    ) -> Any:
        """
        모델명을 기반으로 분류기를 선택
        :param classifier_model: 분류기 알고리즘 이름
        :param classifier_name: 분류기 모델 파일 경로
        :param classifier_output: 출력 노드 이름
        :param classifier_input: 입력 노드 이름 (선택사항)
        :return: 선택된 분류기
        """
        # 모델을 동적으로 가져옴
        classifier_classes = {
            "TensorflowPredict2D": TensorflowPredict2D,
            "TensorflowPredictVGGish": TensorflowPredictVGGish,
            "TensorflowPredictMusiCNN": TensorflowPredictMusiCNN,
            "TensorflowPredictEffnetDiscogs": TensorflowPredictEffnetDiscogs,
        }

        if classifier_model in classifier_classes:
            classifier_class = classifier_classes[classifier_model]
            # classifier_input이 필요한 경우와 필요하지 않은 경우를 구분
            if classifier_input:
                return classifier_class(
                    graphFilename=classifier_name,
                    input=classifier_input,
                    output=classifier_output,
                )
            else:
                return classifier_class(
                    graphFilename=classifier_name, output=classifier_output
                )
        else:
            raise ValueError(f"지원되지 않는 분류기명: {classifier_model}")

    def _get_metadata(self, json_file: str) -> Dict[str, Any]:
        """
        JSON 메타데이터 파일을 읽고 파싱하는 공통 함수.
        :param json_file: JSON 메타데이터 파일 경로
        :return: 파싱된 메타데이터 딕셔너리
        """
        model_path = str(self.models_dir / json_file)

        with open(model_path, "r") as f:
            return json.load(f)

    def _common_predict(
        self, json_file: str, label_prefix: str = "", input_required: bool = False
    ) -> Dict[str, float]:
        """
        공통 prediction 로직을 처리하는 함수.
        :param json_file: JSON 메타데이터 파일 경로
        :param label_prefix: 결과 라벨에 추가할 접두사
        :param input_required: classifier_input이 필요한지 여부
        :return: 예측 결과 딕셔너리
        """
        metadata = self._get_metadata(json_file)
        link = metadata["link"]
        classifier_name = str(self.models_dir / link.split("/")[-1])
        classifier_model = metadata["inference"]["algorithm"]
        classifier_output = metadata["schema"]["outputs"][0].get("name")
        embedding_model = metadata["inference"]["embedding_model"]["model_name"]

        # input_required가 True인 경우에만 classifier_input 추출
        classifier_input = None
        if input_required and "schema" in metadata and "inputs" in metadata["schema"]:
            classifier_input = metadata["schema"]["inputs"][0].get("name")

        # 모델 파일이 존재하지 않으면 다운로드
        if not os.path.exists(classifier_name):
            self._dowonload_models(link)

        embeddings = self._select_embeddings(embedding_model)
        classifier = self._select_classifier(
            classifier_model=classifier_model,
            classifier_name=classifier_name,
            classifier_output=classifier_output,
            classifier_input=classifier_input,
        )
        predictions = classifier(embeddings)
        mean_predictions = predictions.mean(axis=0)

        pred_classes = {}
        for label, probability in zip(metadata["classes"], mean_predictions):
            label = label_prefix + label
            pred_classes[label] = probability

        return pred_classes

    # Moods and context
    def predict_approachability(self) -> Dict[str, float]:
        """
        * approachability(친근감)
        approachability_2c-discogs-effnet-1.pb             16-May-2023 07:55              514458

        """
        return self._common_predict("approachability_2c-discogs-effnet-1.json")

    def predict_engagement(self):
        """
        * engagement(몰입도)
        engagement_2c-discogs-effnet-1.pb                  16-May-2023 07:55              514458
        """
        return self._common_predict(json_file="engagement_2c-discogs-effnet-1.json")

    def predict_danceability(self):
        """
        * danceability
        danceability-audioset-vggish-1.pb                  16-May-2023 07:55               53658
        danceability-audioset-yamnet-1.pb                  16-May-2023 07:55              412058
        danceability-discogs-effnet-1.pb                   16-May-2023 07:55              514458
        danceability-msd-musicnn-1.pb                      16-May-2023 07:55               82458
        danceability-openl3-music-mel128-emb512-1.pb       16-May-2023 07:55              207258
        """
        return self._common_predict("danceability-audioset-vggish-1.json")

    def predict_aggressive(self) -> Dict[str, float]:
        """
        * Aggressive
        """
        return self._common_predict(json_file="mood_aggressive-audioset-vggish-1.json")

    def predict_happy(self) -> Dict[str, float]:
        """
        * happy

        mood_happy-audioset-vggish-1.pb                    16-May-2023 07:55               53658
        mood_happy-audioset-yamnet-1.pb                    16-May-2023 07:55              412058
        mood_happy-discogs-effnet-1.pb                     16-May-2023 07:55              514458
        mood_happy-msd-musicnn-1.pb                        16-May-2023 07:55               82458
        """
        return self._common_predict(json_file="mood_happy-audioset-vggish-1.json")

    def predict_party(self) -> Dict[str, float]:
        """
        * party
        mood_party-audioset-vggish-1.pb                    16-May-2023 07:55               53658
        mood_party-audioset-yamnet-1.pb                    16-May-2023 07:55              412058
        mood_party-discogs-effnet-1.pb                     16-May-2023 07:55              514458
        mood_party-msd-musicnn-1.pb                        16-May-2023 07:55               82458
        """
        return self._common_predict(json_file="mood_party-audioset-vggish-1.json")

    def predict_relaxed(self) -> Dict[str, float]:
        """
        * Relaxed
        """
        return self._common_predict(json_file="mood_relaxed-audioset-vggish-1.json")

    def predict_sad(self) -> Dict[str, float]:
        """
        * sad
        """
        return self._common_predict(json_file="mood_sad-audioset-vggish-1.json")

    def predict_mirex(self) -> Dict[str, float]:
        """
        * MIREX 감정 분류
        moods_mirex-msd-musicnn
        moods_mirex-audioset-vggish-1
        """
        return self._common_predict(
            json_file="moods_mirex-audioset-vggish-1.json",
            label_prefix="mirex_",
            input_required=True,
        )

    def predict_jamendo_mood_and_theme(self) -> Dict[str, float]:
        """
        * MTG-Jamendo mood and theme
        """

        return self._common_predict(
            json_file="mtg_jamendo_moodtheme-discogs-effnet-1.json"
        )

    # Instrumentation
    def predict_acoustic(self) -> Dict[str, float]:
        """
        * Acoustic
        mood_acoustic-audioset-vggish-1.pb                 16-May-2023 07:55               53658
        mood_acoustic-audioset-yamnet-1.pb                 16-May-2023 07:55              412058
        mood_acoustic-discogs-effnet-1.pb                  16-May-2023 07:55              514458
        mood_acoustic-msd-musicnn-1.pb                     16-May-2023 07:55               82458
        """
        return self._common_predict(json_file="mood_acoustic-audioset-vggish-1.json")

    def predict_electronic(self) -> Dict[str, float]:
        """
        * Electronic
        """
        return self._common_predict(json_file="mood_electronic-audioset-vggish-1.json")

    def predict_voice_instrumental(self) -> Dict[str, float]:
        """
        * Voice/instrumental
        """
        return self._common_predict(
            json_file="voice_instrumental-audioset-vggish-1.json"
        )

    def predict_gender(self) -> Dict[str, float]:
        """
        * Voice gender
        """
        return self._common_predict(json_file="gender-audioset-vggish-1.json")

    def predict_timber(self) -> Dict[str, float]:
        """
        * Timbre
        timbre-discogs-effnet-1.pb                         16-May-2023 07:55              514458
        """
        return self._common_predict(json_file="timbre-discogs-effnet-1.json")

    def predict_nsynth_acoustic_electronic(self) -> Dict[str, float]:
        """
        * Nsynth acoustic/electronic
        Classification of monophonic sources into acoustic or electronic origin using the Nsynth dataset
        """
        return self._common_predict(
            json_file="nsynth_acoustic_electronic-discogs-effnet-1.json",
            label_prefix="nsynth_",
        )

    def predict_nsynth_bright_dark(self) -> Dict[str, float]:
        """
        * Nsynth bright/dark
        Classification of monophonic sources by timbre color using the Nsynth dataset
        """
        return self._common_predict(
            json_file="nsynth_bright_dark-discogs-effnet-1.json", label_prefix="nsynth_"
        )

    def predict_nsynth_reverb(self) -> Dict[str, float]:
        """
        * Nsynth reverb
        Detection of reverb in monophonic sources using the Nsynth dataset
        """
        return self._common_predict(
            json_file="nsynth_reverb-discogs-effnet-1.json", label_prefix="nsynth_"
        )

    def predict_tonality(self) -> Dict[str, float]:
        """
        * Tonal/atonal
        """
        return self._common_predict(json_file="tonal_atonal-audioset-vggish-1.json")

    def predict_arousal_valence_deam(self) -> Dict[str, float]:
        """
        DEAM (Database for Emotion Analysis using Music) 데이터셋으로 학습된 모델.
        1,802개 음악에 대해 연속적으로 주석된 감정가(valence)와 각성도(arousal) 값을
        기반으로 VGGish 임베딩을 통해 음악의 감정적 특성을 예측.
        """
        return self._common_predict(
            json_file="deam-audioset-vggish-2.json", label_prefix="deam_"
        )

    def predict_arousal_valence_muse(self) -> Dict[str, float]:
        """
        MuSe (Multimodal Sentiment Analysis in Real-life Media) 데이터셋으로 학습된 모델.
        스트레스 상황에서의 감정 표현과 생리적 반응을 분석하여 각성도(arousal)와
        감정가(valence)를 예측. 실제 환경에서의 멀티모달 감정 인식에 특화됨.
        """
        return self._common_predict(
            json_file="muse-audioset-vggish-2.json", label_prefix="muse_"
        )

    def predict_tempo(self) -> Dict[str, float]:
        """
        * Tempo
        """
        from essentia.standard import TempoCNN  # type: ignore

        json_file = "deepsquare-k16-3.json"

        metadata = self._get_metadata(json_file)
        link = metadata["link"]
        classifier_name = str(self.models_dir / link.split("/")[-1])

        # 모델 파일이 존재하지 않으면 다운로드
        if not os.path.exists(classifier_name):
            self._dowonload_models(link)

        audio = MonoLoader(
            filename=self.audio_file, sampleRate=11025, resampleQuality=4
        )()
        classifier = TempoCNN(graphFilename=classifier_name)
        global_tempo, local_tempo, local_tempo_probabilities = classifier(audio)

        return {
            "avg_tempo": global_tempo,
            # "local_tempo": local_tempo,
            # "local_tempo_probabilities": local_tempo_probabilities.tolist()
        }

    def predict_dynamic_complexity_loudness(self) -> Dict[str, float]:
        """
        * Dynamic complexity and loudness
        음향 신호의 동적 복잡도(dynamic complexity)와 라우드니스 측정.

        전체 라우드니스 레벨에서 벗어나는 평균 절대 편차를 dB 스케일로 계산하여
        라우드니스 변동량을 평가합니다. 트랙의 시작과 끝에 있는 묵음을 제외하여
        정확도를 향상시킵니다. 음악의 동적 범위와 복잡성 분석에 활용됩니다.
        """
        from essentia.standard import DynamicComplexity  # type: ignore

        audio = MonoLoader(filename=self.audio_file, sampleRate=44100)()
        classifier = DynamicComplexity()
        dynamic_complexity, loudness = classifier(audio)

        return {"dynamic_complexity": dynamic_complexity, "avg_loudness": loudness}

    def predict_dissonance(self) -> Dict[str, float]:
        """
        * Dissonance
        스펙트럼 피크 분석을 통한 음향 불협화도(dissonance) 측정.

        스펙트럼 피크들 간의 지각적 거칠음(perceptual roughness)을 분석하여 불협화도를 계산합니다.
        Plomp & Levelt의 음조 협화도 연구를 기반으로 한 dissonance curve를 사용하여
        총 불협화도를 추정합니다. 0(완전 협화)부터 1(완전 불협화) 범위의 값을 출력합니다.
        """
        from essentia.standard import (
            Windowing,  # type: ignore
            Spectrum,  # type: ignore
            SpectralPeaks,  # type: ignore
            FrameGenerator,  # type: ignore
            Dissonance,  # type: ignore
        )

        audio = MonoLoader(filename=self.audio_file, sampleRate=44100)()

        # Frame Processing
        windowing = Windowing(type="blackmanharris92")  # 권장 윈도우
        spectrum = Spectrum()
        spectral_peaks = SpectralPeaks(
            magnitudeThreshold=0.00001,  # dB 임계값 설정
            minFrequency=20,  # 최소 주파수
            maxFrequency=8000,  # 최대 주파수
            maxPeaks=100,  # 최대 피크 수
            orderBy="frequency",  # 주파수 순 정렬 (중요!)
        )

        classifier = Dissonance()

        # Frame-by-frame processing
        for frame in FrameGenerator(audio, frameSize=2048, hopSize=1024):
            # Windowing
            windowed_frame = windowing(frame)

            # Spectrum computation
            spectrum_mag = spectrum(windowed_frame)

            # Extract spectral peaks (이것이 핵심!)
            frequencies, magnitudes = spectral_peaks(spectrum_mag)

            # Compute dissonance (frequencies와 magnitudes 입력)
            if len(frequencies) > 1:  # 최소 2개 피크 필요
                dissonance = classifier(frequencies, magnitudes)

        return {"dissonance": dissonance}

    def predict_all(
        self, exclude_methods: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        모든 predict 함수의 결과를 하나의 dictionary로 병합

        :param exclude_methods: 제외할 메서드 이름 리스트 (예: ['predict_tempo', 'predict_dissonance'])
        :return: 모든 예측 결과가 병합된 딕셔너리
        """

        # 모든 predict_ 메서드 자동 검색
        predict_methods = [
            method
            for method in dir(self)
            if method.startswith("predict_") and callable(getattr(self, method))
        ]

        # predict_all 자신은 제외
        predict_methods = [m for m in predict_methods if m != "predict_all"]

        # 제외할 메서드 필터링
        if exclude_methods:
            predict_methods = [m for m in predict_methods if m not in exclude_methods]

        results = {}
        total_methods = len(predict_methods)

        for i, method_name in enumerate(predict_methods, 1):
            try:
                print(f"[{i}/{total_methods}] 실행 중: {method_name}")
                method = getattr(self, method_name)
                result = method()
                results.update(result)  # 딕셔너리 병합
            except Exception as e:
                print(f"{method_name} 실행 실패: {e}")
                continue

        print(f"총 {len(results)}개 예측 결과 생성 완료")
        return results
