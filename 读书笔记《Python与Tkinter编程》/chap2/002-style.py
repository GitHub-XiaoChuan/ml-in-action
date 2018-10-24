from tkinter import *

class Main():
    def __init__(self, root):
        for relief in [RAISED, SUNKEN, FLAT, RIDGE, GROOVE, SOLID]:
            f = Frame(root, borderwidth=2, relief=relief)
            Label(f, text=relief, width=10).pack(side=LEFT)
            f.pack(side=LEFT, padx=5, pady=5)


if __name__ == '__main__':
    root = Tk()
    tool = Main(root)
    root.mainloop()