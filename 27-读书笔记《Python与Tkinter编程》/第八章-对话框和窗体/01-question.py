from tkinter import *
from tkinter.messagebox import *
import Pmw

# http://pmw.sourceforge.net/doc/EntryField.html
# 标准的对话框

class App:
    def __init__(self, master):
        self.result = Pmw.EntryField(master,
                                     entry_width=8,
                                     value='',
                                     label_text='Returned value:',
                                     labelpos=W,
                                     labelmargin=1)
        self.result.pack(padx=15, pady=15)
root = Tk()
question = App(root)

# 第一个参数表示标题
# 第二个参数表示信息

# 下面出现两个选项，选项会设置到对应的entry中
button = askquestion("Question:",
                     "Oh Dear, did somebody say mattress to Mr Lambert?",
                     default=NO)
question.result.setentry(button)
root.mainloop()
