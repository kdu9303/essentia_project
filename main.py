from rich import print
from classifiers import AudioClassifier

def main():
    audio_file_path = "audio/Love Me Never Ending - Everet Almond.mp3"
    mood_classifier = AudioClassifier(audio_file_path)
    
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
    data = mood_classifier.predict_dissonance()
    print("Data:", data)
    

if __name__ == "__main__":
    main()
