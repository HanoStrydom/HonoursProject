from tkinter import *
from main import faceExpression

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
    root.destroy()
    try:
        faceExpression(int(myVar))
    except:
        print("Please enter a valid time!")
    
    
    
#Button that will run the program
myButton = Button(root, text="Enter", padx=50, command=myClick)
myButton.pack()

root.mainloop()