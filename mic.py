import speech_recognition as sr

recognizer = sr.Recognizer()
recognizer.energy_threshold = 300
mic = sr.Microphone()

with mic as source:
    print("katakan sesuatu\n")
    recognizer.adjust_for_ambient_noise(source)
    audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio, language='id-ID')
        print(text)
    except sr.UnknownValueError:
        print("Suara tidak terdengar, Coba lagi\n")
    except sr.RequestError:
        print("Microphone ada gangguan\n")