from pydub import AudioSegment
audio = AudioSegment.from_file("ouput.wav")

audio=audio +6

audio = audio * 2

audio=audio.fade_in(2000)

audio.export("mashup.wav", format="mp3")

audio=AudioSegment.from_file("mashup.wav")
print("done")