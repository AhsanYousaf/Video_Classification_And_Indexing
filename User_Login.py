from tkinter import *
from os import system
from tkinter import messagebox
import pymysql



class Login:
    def __init__(self, root):
        self.root = root
        self.root.geometry("500x500")
        self.root.resizable(False,False)
        self.root.configure(background="black")
        self.root.title("login")

        title = Label(root, text="*-*-* Login *-*-*", width=20, font=("bold", 20), bg="black", fg='grey')
        title.place(x=90, y=100)

        username = Label(root, text="UserName", width=20, font=("bold", 10), bg="black", fg='grey')
        username.place(x=80, y=200)
        self.txt_username = Entry(root)
        self.txt_username.place(x=220, y=200)

        Password = Label(root, text="Password", width=20, font=("bold", 10), bg="black", fg='grey')
        Password.place(x=80, y=250)

        self.txt_pwd = Entry(root,show="*")
        self.txt_pwd.place(x=220, y=250)

        btn_submit=Button(root, text='Login', width=8, bg='grey', command=self.DB_Conactivity).place(x=220, y=300)
    def DB_Conactivity(self):
         if self.txt_username.get() == "" or self.txt_pwd.get() == "":
           messagebox.showerror("Error", "All fields are Required", parent=self.root)

         else:

            try:
                connection = pymysql.connect(host="localhost", user="root", password="", database="db_connectivity")
                cursor = connection.cursor()
                cursor.execute("select * from userdb where username=%s AND password=%s",
                               (self.txt_username.get(),
                                self.txt_pwd.get(),
                                ))

                results = cursor.fetchall()
                if results:
                    for i in results:

                        messagebox.showinfo("Success", "Successfuly Login", parent=self.root)
                        root.destroy()
                        system('Main_App.py')

                        break
                else:
                         messagebox.showwarning("warning", "username or password is invalid", parent=self.root)
                connection.commit()
                connection.close()


            except Exception as es:
                messagebox.showerror("Error", f"Error due to:{str(es)}", parent=self.root)

root = Tk()
obj=Login(root)
root.mainloop()