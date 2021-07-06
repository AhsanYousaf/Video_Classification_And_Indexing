from tkinter import *
from tkinter import messagebox
import pymysql
from os import system


class Signup:
    def __init__(self, root):
        self.root = root
        self.root.geometry("500x500")
        self.root.resizable(False, False)
        self.root.configure(background="lightgrey")
        self.root.title("Signup")

        frame_signup=Frame(root,bg='#078fc9')
        frame_signup.place(x=50, y=20, width=400, height=55)
        title = Label(text="Signup", width=10, font=('Arial Rounded MT Bold', 25) ,bg='#078fc9' ,fg='white')
        title.place(x=140, y=25)

        frame_entry=Frame(root,bg='white')
        frame_entry.place(x=50,y=80,width=400,height=380)

        name = Label(root, text="Name:", width=10, font=('goudy old style', 15,'bold'), bg="white", fg='black')
        name.place(x=100, y=100)
        self.txt_name = Entry(root,font=('times new roman',15),bg='lightgrey')
        self.txt_name.place(x=130, y=130,width=270,height=30)

        Gender = Label(root, text="Gender", width=10, font=('goudy old style', 15,'bold'), bg="white", fg='black')
        Gender.place(x=105, y=170)
        self.gender = StringVar()
        self.gender.set("Female")
        self.btn_male = Radiobutton(root, text="Male", padx=5, variable=self.gender, value='Male', bg="white",font=(5),
                                    fg='black').place(x=230, y=170)
        self.btn_female = Radiobutton(root, text="Female", padx=20, variable=self.gender, value='Female', bg="white",font=(5),
                                      fg='black').place(x=300, y=170)

        username = Label(root, text="User Name", width=10, font=('goudy old style', 15,'bold'), bg="white", fg='Black')
        username.place(x=120, y=200)
        self.txt_username = Entry(root,font=('times new roman',15),bg='lightgrey')
        self.txt_username.place(x=130, y=230,width=270,height=30)

        password = Label(root, text="Password", width=10, font=('goudy old style', 15,'bold'), bg="white", fg='black')
        password.place(x=115, y=270)
        self.txt_password = Entry(root, show="*",font=('times new roman',15),bg='lightgrey')
        self.txt_password.place(x=130, y=300,width=270,height=30)

        conpass = Label(root, text="Conform Password", width=20, font=('goudy old style', 15,'bold'), bg="white", fg='black')
        conpass.place(x=98, y=340)
        self.txt_conpass = Entry(root, show="*",font=('times new roman',15),bg='lightgrey')
        self.txt_conpass.place(x=130, y=370,width=270,height=30)
        btn_submit = Button(root, text='SUBMIT', width=8, bg='#078fc9', fg='white',font=('goudy old style', 15,'bold') ,command=self.DB_Conactivity).place(
            x=130, y=410, height=35,width=270)

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