from tkinter import *
from os import system
from tkinter import messagebox
import pymysql

class Login:
    def __init__(self, root):
        self.root = root
        self.root.geometry("500x500")
        self.root.resizable(False,False)
        self.root.configure(background="lightgrey")
        self.root.title("login")

        frame_login=Frame(root,bg='#078fc9')
        frame_login.place(x=50, y=50, width=400, height=55)
        title = Label(text="Login", width=10, font=('Arial Rounded MT Bold', 25) ,bg='#078fc9' ,fg='white')
        title.place(x=140, y=50)

        frame_entry=Frame(root,bg='white')
        frame_entry.place(x=50,y=110,width=400,height=340)

        username = Label(root, text="User Name:", width=20, font=('goudy old style', 15,'bold'), bg="white", fg='black')
        username.place(x=60, y=150)
        self.txt_username = Entry(root,font=('times new roman',15),bg='lightgrey')
        self.txt_username.place(x=120, y=180,width=270,height=40)

        Password = Label(root, text="Password:", width=20, font=('goudy old style', 15,'bold'), bg="white", fg='Black')
        Password.place(x=50, y=240)

        self.txt_pwd = Entry(root,show="*",font=('times new roman',15),bg='lightgrey')
        self.txt_pwd.place(x=120, y=270,width=270,height=40)

        btn_submit=Button(root, text='LOGIN', width=8, bg='#078fc9',fg='white',font=('goudy old style', 15,'bold') ,command=self.DB_Conactivity).place(x=120, y=350,height=40,width=270)
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