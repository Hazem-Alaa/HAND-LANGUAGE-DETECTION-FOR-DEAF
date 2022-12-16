import pyttsx3

input_1 = input("wht ?"
                "")
labels = ['i love you ', 'hello ', 'good', 'yes']

if 'i love you'  == labels[0]:
    text = pyttsx3.init()
    text.say("i love you ")
    text.runAndWait()

if input_1 == labels[1]:
    text = pyttsx3.init()
    text.say("hello ")
    text.runAndWait()
if input_1 == labels[2]:
    text = pyttsx3.init()
    text.say("good ")
    text.runAndWait()
if input_1 == labels[3]:
    text = pyttsx3.init()
    text.say("yes")
    text.runAndWait()


