import matplotlib.pyplot as plt
import csv

#Function that will read the .csv file and display on a graph
def showResults():
    x = []
    y = []
    try:
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
    except:
        print("File not found!")
    #Plots the data
    plt.plot(x,y, label='Confused')
    plt.xlabel('Time')
    plt.ylabel('Emotion')
    plt.title('General Emotion')
    plt.legend()
    plt.show()
