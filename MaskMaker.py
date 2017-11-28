from Tkinter import *
from tkFileDialog import askopenfilename, asksaveasfilename
from PIL import ImageTk, Image, ImageDraw, ImageChops

##########################################################
#Mask making Tkinter program
##########
class MaskMaker(object):
    def __init__(self, master):
        #Initialising the window
        self.master = master
        self.master.title("MaskMaker")
        self.master.bind("<Key>", self.handle_key)

        ##########
        #Declaring attributes
        self.mask_draw = None#ImageDraw handle on the mask image
        self.current_view = None #View state
        self.dst_extension = None #Extension to be recommended for the mask
        #Raw images
        self.src = None
        self.resized_src = None
        self.mask = None
        self.dst = None
        #Shown images
        self.viewed_src = None
        self.viewed_dst = None
        #tkinter versions of images
        self.src_tk = None
        self.mask_tk = None
        self.dst_tk = None
        #Edition parameters
        self.brush_size = 10
        self.mask_opacity = 0.6
        #Image state
        self.zoom_level = 1


        ########
        #initialising the main view
        self.instructions_label = Label(self.master, text="Please create a session")
        self.instructions_label.pack()
        self.image_label = Label(self.master)
        self.image_label.pack()


        #########
        #Initialising the menu
        self.menu = Menu(self.master)
        #File submenu
        self.file_menu = Menu(self.menu, tearoff=0)
        self.file_menu.add_command(label="New Session", command=self.new)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Save Mask As", command=self.save_mask)
        self.file_menu.add_command(label="Save Adjusted Source As", command=self.save_src)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.master.quit)
        self.menu.add_cascade(label="File", menu=self.file_menu)
        #View Submenu
        self.view_menu = Menu(self.menu, tearoff=0)
        self.view_menu.add_command(label="View Source (1)", command=self.see_src)
        self.view_menu.add_command(label="View Mask (2)", command=self.see_mask)
        self.view_menu.add_command(label="View Destination (3)", command=self.see_dst)
        self.menu.add_cascade(label="View", menu=self.view_menu)
        #Setting menu to main window
        self.master.config(menu=self.menu)

    #Represents the popup shown on a new session to pick source and destination images
    class SessionPopup(object):
        def __init__(self, confirmation_callback):
            self.confirmation_callback = confirmation_callback
            self.src_filename = None
            self.dst_filename = None

            self.top = Toplevel()
            self.top.title("Please select a source and a destination image")

            src_frame = Frame(self.top)
            src_frame.pack()
            src_label = Label(src_frame, text="Source Image : ")
            src_label.pack(side=LEFT)
            self.src_value = Label(src_frame, text="None selected")
            self.src_value.pack(side=LEFT)
            src_button = Button(src_frame, text="Pick", command=self.pick_src)
            src_button.pack(side=LEFT)

            dst_frame = Frame(self.top)
            dst_frame.pack()
            dst_label = Label(dst_frame, text="Destination Image : ")
            dst_label.pack(side=LEFT)
            self.dst_value = Label(dst_frame, text="None selected")
            self.dst_value.pack(side=LEFT)
            dst_button = Button(dst_frame, text="Pick", command=self.pick_dst)
            dst_button.pack(side=LEFT)

            button_frame = Frame(self.top)
            button_frame.pack(side=BOTTOM)
            cancel = Button(button_frame, text="Cancel", command=self.top.destroy)
            cancel.pack(side=LEFT)
            self.confirm_btn = Button(button_frame, text="Confirm", state=DISABLED, command=self.confirm)
            self.confirm_btn.pack(side=LEFT)

        def confirm(self):
            self.top.destroy()
            self.confirmation_callback(self.src_filename, self.dst_filename, self)

        def pick_src(self):
            filename = askopenfilename()
            if filename == '':
                return
            self.src_filename = filename
            self.src_value.configure(text=self.src_filename)
            self.activate_confirm()

        def pick_dst(self):
            filename = askopenfilename()
            if filename == '':
                return
            self.dst_filename = filename
            self.dst_value.configure(text=self.dst_filename)
            self.activate_confirm()

        def activate_confirm(self):
            if self.src_filename is not None and self.dst_filename is not None:
                self.confirm_btn.configure(state=NORMAL)


    #Shows a popup to initialise a Session
    def new(self):
        self.SessionPopup(self.init_session)

    #Initialise a session with a source and a destination image
    def init_session(self, src_filename, dst_filename, popup):
        # Make sure the SessionPopup is deleted
        del popup

        #Open the images and keep them in memory : we'll need them to blend the mask
        self.src = Image.open(src_filename)
        self.dst = Image.open(dst_filename)

        #Make sure that the images are of the same color mode (RGB, RGBA, L, etc)
        if self.src.mode != self.dst.mode:
            self.instructions_label.configure(text="Please select two image using the same same color mode.\n Ex: RGB, RGBA, L, etc.")
            return

        #Take note of the destination's extension, we'll recommend it when saving
        parts = dst_filename.split('.')
        self.dst_extension = "." + parts[len(parts)-1]

        #Create resized source image to fit destination size
        self.zoom_level = 1.0
        self.resized_src = self.resize_src(self.src)

        #Create an image in memory for the mask and also initialise an ImageDraw to edit it
        self.mask = Image.new(mode=self.resized_src.mode, size=self.resized_src.size, color=self.get_color(0))
        self.mask_draw = ImageDraw.Draw(self.mask)

        #Default view : show the source image
        self.see_src()
        self.current_view = '1'
        self.instructions_label.configure(text="Left click and drag to add draw to the mask.\nRight click and drag to remove from the mask.\nMove source and mask with w,a,s,d(1px) or W,A,S,D(10px)\nUse 1,2,3 to view source, mask, destination")

        #Record left click drag to add a brush patch to the mask, and right click drag to remove it from the mask
        self.image_label.bind('<B1-Motion>', lambda e: self.color_patch(e, 255))
        self.image_label.bind('<B2-Motion>', lambda e: self.color_patch(e, 0))

    #Sets the src image to the same size as the destination image
    def resize_src(self, src, zoom_level = 1.0):
        #We store the resized image in another variable to still be able to manipulate
        #the original one for offsets
        new_size = (int(zoom_level * src.width), int(zoom_level * src.height))
        resized_src = src.resize(new_size)

        should_crop = False
        crop_size = [resized_src.width, resized_src.height]

        #We check if we need to crop horizontally to fit destination size
        if self.dst.width < resized_src.width:
            should_crop = True
            crop_size[0] = self.dst.width

        #We check if we need to crop or extend vertically to fit destination size
        if self.dst.height < resized_src.height:
            should_crop = True
            crop_size[1] = self.dst.height

        #Crop image if required
        if should_crop:
            resized_src = resized_src.crop((0,0,crop_size[0],crop_size[1]))

        should_extend = False
        extend_size = [resized_src.width, resized_src.height]

        #We check if we need to extend horizontally to fit destination size
        if self.dst.width > resized_src.width:
            should_extend = True
            extend_size[0] = self.dst.width

        #We check if we need to extend vertically to fit destination size
        if self.dst.height > resized_src.height:
            should_extend = True
            extend_size[1] = self.dst.height

        #Extend image if required
        if should_extend:
            temp = Image.new(mode= resized_src.mode, size=(extend_size[0], extend_size[1]), color=self.get_color(0))
            temp.paste(resized_src)
            resized_src = temp

        return resized_src

    #Offsets the resized source and the mask in a direction
    def offset_images(self, x_offset=0, y_offset=0):
        if self.resized_src is None or self.mask is None:
            return

        self.resized_src = ImageChops.offset(self.resized_src,x_offset, y_offset)
        self.mask = ImageChops.offset(self.mask, x_offset, y_offset)
        self.mask_draw = ImageDraw.Draw(self.mask) #update draw handle

        self.update_view()

    def zoom_images(self, zoom_factor):
        if self.src is None or self.mask is None:
            return
        self.zoom_level = self.zoom_level * zoom_factor
        self.resized_src = self.resize_src(self.src, self.zoom_level)
        self.mask = self.resize_src(self.mask, zoom_factor)
        self.mask_draw = ImageDraw.Draw(self.mask) #update draw handle

        self.update_view()


    #Saves the current session's mask
    def save_mask(self):
        if self.mask is None:
            return
        filename = asksaveasfilename(defaultextension=self.dst_extension)
        if filename == '':
            return
        self.mask.save(filename)

    #Saves a modified version of the source image to fit the destination image's
    #size and offseted like shown in the viewer
    def save_src(self):
        if self.resized_src is None:
            return
        filename = asksaveasfilename(defaultextension=self.dst_extension)
        if filename == '':
            return
        self.resized_src.save(filename)

    #Switch view to show the source image with a transparent mask on it
    def see_src(self):
        if self.resized_src is None or self.mask is None:
            return
        self.viewed_src = Image.blend(self.resized_src, self.mask, self.mask_opacity)
        self.src_tk = ImageTk.PhotoImage(self.viewed_src)
        self.image_label.configure(image=self.src_tk)

    #Switch view to show the mask image
    def see_mask(self):
        if self.mask is None:
            return
        self.mask_tk = ImageTk.PhotoImage(self.mask)
        self.image_label.configure(image=self.mask_tk)

    #Switch view to show the destination image with a transparent inverse mask on it
    def see_dst(self):
        if self.dst is None or self.mask is None:
            return
        self.viewed_dst = Image.blend(self.dst, self.mask, self.mask_opacity)
        self.dst_tk = ImageTk.PhotoImage(self.viewed_dst)
        self.image_label.configure(image=self.dst_tk)

    #Refreshes current view with up to date data
    def update_view(self):
        if self.current_view == '1':
            self.see_src()
        elif self.current_view == '2':
            self.see_mask()
        elif self.current_view == '3':
            self.see_dst()

    #Key event handler
    def handle_key(self, event):
        if event.char == '1':
            self.current_view = '1'
            self.see_src()
        elif event.char == '2':
            self.current_view = '2'
            self.see_mask()
        elif event.char == '3':
            self.current_view = '3'
            self.see_dst()
        elif event.char == 'w':
            self.offset_images(y_offset=-1)
        elif event.char == 'a':
            self.offset_images(x_offset=-1)
        elif event.char == 's':
            self.offset_images(y_offset=1)
        elif event.char == 'd':
            self.offset_images(x_offset=1)
        elif event.char == 'W':
            self.offset_images(y_offset=-10)
        elif event.char == 'A':
            self.offset_images(x_offset=-10)
        elif event.char == 'S':
            self.offset_images(y_offset=10)
        elif event.char == 'D':
            self.offset_images(x_offset=10)
        elif event.char == '-':
            self.zoom_images(0.9)
        elif event.char == '+':
            self.zoom_images(1/0.9)

    #Colors a patch of the mask according to a motion event
    def color_patch(self, event, color):
        bounds = [max(0, event.x - self.brush_size),
                  max(0, event.y - self.brush_size),
                  min(self.mask.width, event.x + self.brush_size),
                  min(self.mask.height, event.y + self.brush_size)]
        self.mask_draw.ellipse(bounds, outline=self.get_color(color), fill=self.get_color(color))
        self.update_view()

    #Utility giving the color represented correctly for the color mode used in src and dst
    #|color| param should be a value of 0 to 255
    #Currently, only formats using values from 0 to 255 for each color channel are supported
    def get_color(self, color):
        if self.dst.mode == 'L':
            return color
        elif self.dst.mode == 'RGB':
            return (color,color,color)
        elif self.dst.mode == 'RGBA':
            return (color,color,color,255)

root = Tk()
my_gui = MaskMaker(root)

#Have window be in front when created
root.lift()
root.attributes('-topmost',True)
root.after_idle(root.attributes,'-topmost',False)


root.mainloop()
