import wave

obj = wave.open("01-basics_output.wav","rb")

print("Number of channels: ", obj.getnchannels())
print("Sample width: ",obj.getsampwidth())
print("Framerate: ",obj.getframerate())
print("Number of frames: ",obj.getnframes())
print("Parameters: ",obj.getparams())

time = obj.getnframes()/obj.getframerate()
print("Time: ",time)

frames = obj.readframes(-1)
print(type(frames), type(frames[0]))
print(len(frames)/2)

obj.close()

obj_new = wave.open("01-basics_output_new.wav","wb")
obj_new.setnchannels(1)
obj_new.setsampwidth(2)
obj_new.setframerate(16000.0)

obj_new.writeframes(frames)

obj_new.close()