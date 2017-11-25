from Tkinter import Tk, Menu, Label, Frame
from tkFileDialog import askopenfilename, asksaveasfilename
from PIL import ImageTk, Image, ImageDraw

class MainUI:
    def __init__(self, master):
        self.brush_size = 10
        
        self.master = master
        self.master.title("MainUI")
        self.master.bind("<Key>", self.handle_key)

        self.src = None

        self.menu = Menu(self.master)

        self.file_menu = Menu(self.menu, tearoff=0)
        self.file_menu.add_command(label="New Mask", command=self.new)
        self.file_menu.add_command(label="Save Mask", command=self.save)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.master.quit)
        self.menu.add_cascade(label="File", menu=self.file_menu)

        self.edit_menu = Menu(self.menu, tearoff=0)
        self.edit_menu.add_command(label="See Mask", command=self.see_mask)
        self.edit_menu.add_command(label="See Source", command=self.see_src)
        self.menu.add_cascade(label="Edit", menu=self.edit_menu)

        self.master.config(menu=self.menu)

    def new(self):
        filename = askopenfilename()
        if filename == '':
            return
        self.src = Image.open(filename)
        self.src_tk = ImageTk.PhotoImage(self.src)

        self.mask = Image.new(mode='L', size=self.src.size, color=0)
        self.mask_draw = ImageDraw.Draw(self.mask)

        self.src_panel = Label(self.master, image=self.src_tk)
        self.src_panel.bind('<B1-Motion>', self.select_patch)
        self.src_panel.pack()

    def select_patch(self, event):
        self.mask_draw.ellipse([max(0, event.x - self.brush_size), max(0, event.y - self.brush_size), min(self.mask.width, event.x + self.brush_size), min(self.mask.height, event.y + self.brush_size)], outline=255, fill=255)
        self.see_mask()
    
    def see_mask(self):
        self.mask_tk = ImageTk.PhotoImage(self.mask)
        self.src_panel.configure(image=self.mask_tk)

    def see_src(self):
        self.src_panel.configure(image=self.src_tk)

    def handle_key(self, event):
        if event.char == '1':
            self.see_mask()
        elif event.char == '2':
            self.see_src()

    def save(self):
        filename = asksaveasfilename(defaultextension=".jpg")
        if filename == '':
            return
        self.mask.save(filename, "JPEG")
        
        

root = Tk()
my_gui = MainUI(root)
root.mainloop()
