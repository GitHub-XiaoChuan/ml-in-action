from tkinter import *

class Main():
    def __init__(self, root):
        Label(root, text="I mean, it's a little confusing for me when you say",
              wraplength=150, justify=LEFT).pack(pady=10)
        f1 = Frame(root)
        Label(f1, text="it's not working, we need more!", relief=RAISED).pack(pady=10)
        Label(f1, text="I'm nmot comming out!", relief=SUNKEN).pack(side=LEFT, padx=5)
        f1.pack()

        # f2 = Frame(root)
        # for bitmap, rlf in [('woman', RAISED)]:
        #     Label(f2, bitmap='@bitmaps/%s'%bitmap, relief=rlf).pack(side=LEFT, padx=5)
        # f2.pack()

if __name__ == '__main__':
    root = Tk()
    main = Main(root)
    root.mainloop()