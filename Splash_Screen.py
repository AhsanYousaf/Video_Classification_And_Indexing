from tkinter import *
from os import system


splash_root = Tk()


splash_root.title("Splash Screen")
splash_root.geometry("800x600")

splash_root.configure(background="lightgrey")

def login():
    #splash_root.destroy()
    system('User_Login.py')

def signup():
    #splash_root.destroy()
    system('User_Signup.py')


def main():

    splash_root.destroy()
    root = Tk()


    root.title("Video Classifier and Indexer")
    root.geometry("500x500")
    root.configure(background="lightgray")



    L_to_Con = Label(root, text="Login to Continue", font=("bold", 30,), bg="lightgrey", fg='Black').place(x=100, y=50)
    login_label = Label(root, text="Already a memeber ?", bg="lightgrey",font=('goudy old style', 15), fg="black").place(x=180, y=200)
    Button(root, text='Login', width=25, bg='#078fc9',fg='white',font=('goudy old style', 15,'bold'), command=login).place(x=130, y=230)
    signup_label = Label(root, text="Dont have an account ?", bg="lightgrey",font=('goudy old style', 15), fg="black").place(x=180,y=300)
    Button(root, text='Signup',width=25, bg='#078fc9',fg='white',font=('goudy old style', 15,'bold'), command=signup).place(x=130, y=330)



splash_root.after(1500, main)

mainloop()