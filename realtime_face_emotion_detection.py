"""
Honours Project
@student: NWU - 31597793
@author: Hano Strydom
@email: hanostrydom8@gmail.com
@supervisor: Mnr. Henri van Rensburg
"""

#importing the required libraries
import csv
import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from keras.models import model_from_json
import face_recognition
import datetime
import time
from ast import Lambda
from cgi import test
from tkinter import *
import matplotlib.pyplot as plt


#Python user interface that ask for a time when the user wants to see the general sentiment
root = Tk()
root.title('Lecturer Expression Tool')
root.geometry('420x130+50+50')
root.resizable(False, False)
root.eval('tk::PlaceWindow . center')
myLabel1 = Label(root, text="Welcome Lecturer!",font=('Helvetica', 18, 'bold'))
myLabel1.pack()
myLabel2 = Label(root, text="Please enter a time to keep track (in seconds): ",font=('Helvetica', 15,))
myLabel2.pack()
e = Entry(root, borderwidth=5)
e.pack()
e.get()

myVar = ""

#Function that will run the program when the button is clicked
def myClick():
    myVar = e.get()
    faceExpression(int(myVar))
    
#Button that will run the program
myButton = Button(root, text="Enter", padx=50, command=myClick)
myButton.pack()

#Function that will read the .csv file and display on a graph
def showResults():
    x = []
    y = []
    with open('GeneralEmotion.csv','r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        
        for row in plots:
            if row[1] == " Time":
                continue
            time = row[1]
            #Convert time to dateTime    
            time = time.split(":")
            x.append(str(time[0]) + ":" + str(time[1])+ ":" + str(round(float(time[2]),0)))
            y.append(str(row[2]))

    #Plots the data
    plt.plot(x,y, label='Confused')
    plt.xlabel('Time')
    plt.ylabel('Emotion')
    plt.title('General Emotion')
    plt.legend()
    plt.show()


#The facial expression recognition code that determines the emotion
def faceExpression(myTime):
    root.destroy()
    '''For the seconds tracking emotion'''
    timeConfused = 0
    timeHappy = 0
    timeNeutral = 0
    
    
    #Face expression model initialization
    face_exp_model = model_from_json(open("dataset/facial_expression_model_structure.json", "r").read())
    #load weights into model
    face_exp_model.load_weights('dataset/facial_expression_model_weights.h5')
    #list of emotion labels
    emotions_label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

    webcam_video_stream = cv2.VideoCapture(0)
    all_face_location = []

    '''Sets and inital time'''
    presentDate = datetime.datetime.now()
    initialTime = round(datetime.datetime.timestamp(presentDate))

    #ConCount = 0
    while True:
        ret, current_frame = webcam_video_stream.read()
        
        '''This detects smaller / father faces '''
        #all_face_locations = face_recognition.face_locations(current_frame, number_of_times_to_upsample=2, model="hog")
        
        '''Initialise counters'''
        ConCount = 0
        GoodCount = 0
        NeutralCount = 0
        generalEmotion = ""
        startTime = 0
        newUnixTime = 0
        x = ""

        font = cv2.FONT_HERSHEY_DUPLEX
        all_face_locations = face_recognition.face_locations(current_frame, model="hog")

        for index, current_face_location in enumerate(all_face_locations):
            #plitting the tuple to get the four position values
            top_pos, right_pos, bottom_pos, left_pos = current_face_location

            current_face_image = current_frame[top_pos:bottom_pos,left_pos:right_pos]
            cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)
            
            current_face_image = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY)
            current_face_image = cv2.resize(current_face_image, (48,48))
            img_pixels = image.img_to_array(current_face_image)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            img_pixels /= 255
            
            exp_predictions = face_exp_model.predict(img_pixels)
            max_index = np.argmax(exp_predictions[0])
            emotion_label = emotions_label[max_index]
            
            #Counts the emotions
            if emotion_label == "fear" or emotion_label == "disgust" or emotion_label == "sad" or emotion_label == "angry":
                ConCount = ConCount + 1
                timeConfused = timeConfused + 1

            if emotion_label == "happy" or emotion_label == "surprise" :
                GoodCount = GoodCount + 1
                timeHappy = timeHappy + 1

            if emotion_label == "neutral":
                NeutralCount = NeutralCount + 1
                timeNeutral = timeNeutral + 1

            current_face_image = current_frame[top_pos:bottom_pos,left_pos:right_pos]
            current_face_image = cv2.GaussianBlur(current_face_image, (99,99), 30)
            current_frame[top_pos:bottom_pos,left_pos:right_pos] = current_face_image

            '''Shows individual emotion'''
            #cv2.putText(current_frame, emotion_label, (left_pos,bottom_pos), font, 0.5, (255,255,255),1)
            '''Does not show individual emotion'''
            cv2.putText(current_frame, "student", (left_pos,bottom_pos), font, 0.5, (255,255,255),1)

            totalFaces = index + 1
            percentage = ConCount / totalFaces * 100

            #General Sentiment
            if(ConCount > GoodCount and ConCount > NeutralCount):
                print("General Sentiment is Confused:  " , ConCount)
                print("Percentage of class confused: " + str(percentage) + "%")
            elif(GoodCount > ConCount and GoodCount > ConCount):
                print("General Sentiment is Happy: " , GoodCount)
                print("Percentage of class confused: " + str(percentage) + "%")
            elif(NeutralCount > GoodCount and NeutralCount > ConCount):
                print("General Sentiment is Neutral: " , NeutralCount)
                print("Percentage of class confused: " + str(percentage) + "%")
            elif(GoodCount == ConCount):
                print("General Sentiment is Even: Confused:" , ConCount, " Happy: ", GoodCount)
                print("Percentage of class confused: " + str(percentage) + "%")
            else:
                print("No faces detected!")

            #Get the time
            presentDate = datetime.datetime.now()
            unix_timestamp = datetime.datetime.timestamp(presentDate)
            newUnixTime = round(unix_timestamp)

            #Sets a general sentiment for every 'variable' inserted
            seconds = int(myTime)
            if(newUnixTime == initialTime + seconds):
                initialTime = newUnixTime
                if(timeConfused > timeHappy and timeConfused > timeNeutral):
                    generalEmotion = "Confused"
                    print("Confused: ", timeConfused)
                    print("Happy: ", timeHappy)
                    print("Neutral: ", timeNeutral)
                if(timeHappy > timeConfused and timeHappy > timeNeutral):
                    generalEmotion = "Happy"
                    print("Confused: ", timeConfused)
                    print("Happy: ", timeHappy)
                    print("Neutral: ", timeNeutral)
                if(timeHappy == timeConfused):
                    generalEmotion = "Even"
                    print("Confused: ", timeConfused)
                    print("Happy: ", timeHappy)
                    print("Neutral: ", timeNeutral)
                if(timeNeutral > timeHappy and timeNeutral > timeConfused):
                    generalEmotion = "Neutral"
                    print("Confused: ", timeConfused)
                    print("Happy: ", timeHappy)
                    print("Neutral: ", timeNeutral)
                print("General sentiment for the time selected: " , generalEmotion) 
                
                #Checks whether GeneralEmotion.csv exists
                x = str(presentDate).split(" ")
                if os.path.isfile('GeneralEmotion.csv') == True:
                    with open('GeneralEmotion.csv', 'a') as csvFile:
                        csvFile.write(str(x[0])+ ", " + str(x[1]) + ", " + generalEmotion + '\n')
                    csvFile.close()
                else:
                    print("File Created!")
                    with open('GeneralEmotion.csv', 'w') as csvFile:
                        csvFile.write("Date, Time, General Emotion" + '\n')
                        csvFile.write(str(x[0])+ ", " + str(x[1]) + ", " + generalEmotion + '\n')
                    csvFile.close()
                                    
                timeConfused = 0
                timeHappy = 0
                timeNeutral = 0
    
        #Display the number of confused students
        text =  "Confused: " + str(ConCount)
        cv2.putText(current_frame, text, (30,100), font, 3, 0.5, cv2.LINE_AA)   
        cv2.imshow("Webcam Video ", current_frame)

        #Stops the program with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam_video_stream.release()
    cv2.destroyAllWindows()
    showResults()
root.mainloop()
