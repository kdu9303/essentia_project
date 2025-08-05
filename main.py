from typing import List
from rich import print
from classifiers import AudioClassifier
from utils import get_audio_files, cut_audio_file


def preprocess_audio_files() -> None:
    """
    오디오 파일을 전처리하고, 각 파일에 대해 AudioClassifier를 초기화합니다.
    30, 45, 60초, 전구간 별 자르기
    dataframe 저장
    
    """
    audio_files: List = get_audio_files("audio/", is_path_form=True, extensions=[".mp3"])
    
    for audio in audio_files:
        print(f"현재 Process 중인 파일: {audio}")
        
        
        



def main():
    audio_files: List = get_audio_files("audio/", is_path_form=True, extensions=[".mp3"])
    
    for audio in audio_files:
        print(f"현재 Process 중인 파일: {audio}")
        classifier = AudioClassifier(audio)
    
    # Example usage of the classifier methods
    # danceability = mood_classifier.predict_danceability()
    # approachability = mood_classifier.predict_approachability()
    # engagement = mood_classifier.predict_engagement()
    # print("Danceability:", danceability)
    # print("Approachability:", approachability)
    # print("Engagement:", engagement)
    
    # arousal_valence_muse = mood_classifier.predict_arousal_valence_muse()
    # print("Arousal-Valence Muse:", arousal_valence_muse)
    # arousal_valence_deam = mood_classifier.predict_arousal_valence_deam()
    # print("Arousal-Valence DEAM:", arousal_valence_deam)
    # mirex = mood_classifier.predict_mirex()
    # print("Mirex:", mirex)
    # data = mood_classifier.predict_dissonance()
    # print("Data:", data)
    

if __name__ == "__main__":
    main()
