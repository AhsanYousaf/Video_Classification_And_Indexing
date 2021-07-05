from tkinter import *
from os import system

splash_root = Tk()


splash_root.title("Splash Screem")
splash_root.geometry("800x600")
splash_root.configure(background="black")

background_image = PhotoImage(file="splash-01.png")

background = Label(splash_root, image=background_image, bd=0)
background.pack()

def login():

    splash_root.destroy()
    system('User_Login.py')

def signup():
    splash_root.destroy()
    system('User_Signup.py')


def main():

    splash_root.destroy()
    root = Tk()


    root.title("Video Classifier and Indexer")
    root.geometry("500x500")
    root.configure(background="black")



    L_to_Con = Label(root, text="Login to Continue", font=("bold", 30,), bg="black", fg='grey').place(x=100, y=50)
    login_label = Label(root, text="Already a memeber ?", bg="black", fg="grey").place(x=200, y=200)
    Button(root, text='Login', width=20, bg='grey', command=login).place(x=180, y=250)
    signup_label = Label(root, text="Dont have an account ?", bg="black", fg="grey").place(x=190,y=300)
    Button(root, text='Signup', width=20, bg='grey', command=signup).place(x=180, y=350)



splash_root.after(1500, main)


mainloop()