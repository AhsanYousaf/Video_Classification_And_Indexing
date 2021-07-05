from tkinter import *
from tkinter import messagebox
import pymysql
from os import system


class Signup:
    def __init__(self, root):
        self.root = root
        self.root.geometry("500x500")
        self.root.resizable(False, False)
        self.root.configure(background="black")
        self.root.title("Signup")

        title = Label(root, text="*-*-* Signup *-*-* ", width=20, font=("bold", 20), bg="black", fg='grey')
        title.place(x=90, y=53)

        name = Label(root, text="Name", width=20, font=("bold", 10), bg="black", fg='grey')
        name.place(x=48, y=130)
        self.txt_name = Entry(root)
        self.txt_name.place(x=240, y=130)

        Gender = Label(root, text="Gender", width=20, font=("bold", 10), bg="black", fg='grey')
        Gender.place(x=50, y=180)
        self.gender = StringVar()
        self.gender.set("Female")
        self.btn_male = Radiobutton(root, text="Male", padx=5, variable=self.gender, value='Male', bg="black",
                                    fg='grey').place(x=235, y=180)
        self.btn_female = Radiobutton(root, text="Female", padx=20, variable=self.gender, value='Female', bg="black",
                                      fg='grey').place(x=290, y=180)

        username = Label(root, text="UserName", width=20, font=("bold", 10), bg="black", fg='grey')
        username.place(x=56, y=230)
        self.txt_username = Entry(root)
        self.txt_username.place(x=240, y=230)

        password = Label(root, text="Password", width=20, font=("bold", 10), bg="black", fg='grey')
        password.place(x=56, y=280)
        self.txt_password = Entry(root, show="*")
        self.txt_password.place(x=240, y=280)

        conpass = Label(root, text="Conform Password", width=20, font=("bold", 10), bg="black", fg='grey')
        conpass.place(x=82, y=330)
        self.txt_conpass = Entry(root, show="*")
        self.txt_conpass.place(x=240, y=330)
        btn_submit = Button(root, text='Submit', width=8, bg='grey', fg='black', command=self.DB_Conactivity).place(
            x=220, y=380)

    def DB_Conactivity(self):
        if self.txt_name.get() == "" or self.txt_username.get() == "" or self.gender.get() == "Select" or self.txt_password.get() == "" or self.txt_conpass.get() == "":
            messagebox.showwarning("Error", "All fields are Required", parent=self.root)
        elif self.txt_password.get() != self.txt_conpass.get():
            messagebox.showwarning("Error", "Password & confirm password should be same", parent=self.root)

        else:

            try:
                connection = pymysql.connect(host="localhost", user="root", password="", database="db_connectivity")
                cursor = connection.cursor()

                cursor.execute("select * from userdb where username=%s ",
                               (self.txt_username.get()
                                ))

                results = cursor.fetchall()
                if results:
                    for i in results:
                        results == self.txt_username.get()
                        messagebox.showwarning("warning", "User name already exist", parent=self.root)


                else:
                    cursor.execute("insert into userdb(Name,Gender,UserName,Password) values(%s,%s,%s,%s)",
                                   (self.txt_name.get(),
                                    self.gender.get(),
                                    self.txt_username.get(),
                                    self.txt_password.get(),

                                    ))
                    messagebox.showinfo("Success", "Successfuly Signup", parent=self.root)
                    root.destroy()
                    system('User_Login.py')
                connection.commit()
                connection.close()


            except Exception as es:
                messagebox.showerror("Error", f"Error due to:{str(es)}", parent=self.root)



root = Tk()
obj = Signup(root)
root.mainloop()