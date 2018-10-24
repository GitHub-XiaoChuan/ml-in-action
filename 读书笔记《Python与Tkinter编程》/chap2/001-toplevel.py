from tkinter import *

class Main(Frame):
    def __init__(self):
        Frame.__init__(self)

        root = Tk()
        #root.option_readfile('optionDB')
        root.title('TopLevel')

        Label(root, text='this is the main(default) toplevel').pack(pady=10)

        t1 = Toplevel(root)
        Label(t1, text='This is a child of root').pack(padx=10, pady=10)

        t2 = Toplevel(root)
        Label(t2, text='this is a transient window of root').pack(padx=10, pady=10)
        t2.transient(root)

        t3 = Toplevel(root, borderwidth=5, bg='blue')
        Label(t3, text='no wm decorations', bg='blue', fg='white').pack(padx=10, pady=10)
        t3.overrideredirect(1)
        t3.geometry('200x70+150+150')

if __name__ == '__main__':
    Main().mainloop()