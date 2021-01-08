from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import object_tracker
import os
from time import sleep

class Root(Tk):

    def __init__(self):
        super(Root, self).__init__()
        self.title("Recognize people and color")
        self.minsize(500, 400)
        self.labelFrame = LabelFrame(self, text="Otwórz plik")
        self.label_file = ttk.Label(self.labelFrame, text="")
        self.labelFrame.grid(column=3, row=1, padx=20, pady=20)

        self.button = ttk.Button(self.labelFrame, text="Wybierz film",
                                 command=lambda: [self.file_search(), self.button_cb()])
        self.button.grid(column=1, row=1)

    #GUI buttons
    def button_cb(self):
        if self.filename != '':
            self.button = ttk.Button(self.labelFrame, text="Uruchom program",
                                     command=lambda: [self.run(),self.button_open(),self.button_open_txt()])
            self.button.grid(column=1, row=9)


    def button_open(self):
        if self.filename != '':
            self.button = ttk.Button(self.labelFrame, text="Odtworz film",
                                     command=lambda: [object_tracker.player("{}".format(self.output_filename))])
            self.button.grid(column=1, row=10)
            self.label_file.grid(column=1, row=12)
            self.label_file.configure(
                text='Film zapisano do: ' + self.output_filename +'\nLogi zapisano do: '+ self.output_filename_log)

    def button_open_txt(self):
        if self.filename != '':
            self.button = ttk.Button(self.labelFrame, text="Odtworz plik z logami",
                                     command=lambda: [object_tracker.opentxt("{}".format(self.output_filename_log))])
            self.button.grid(column=1, row=11)


    #tracker initialization
    def run(self):
        if self.filename != '':
            object_tracker.start_tracker(self.filename, self.output_filename, self.output_filename_log)
            self.label_file.grid(column=1, row=3)
            self.label_file.configure(text='Trwa przetwarzanie...')
            self.progress_bar = ttk.Progressbar(self.labelFrame, orient=HORIZONTAL, length=500, mode='determinate')
            self.progress_bar.grid(column=1, row=5)
            progress = 0
            while progress < 1:
                sleep(1)
                progress = object_tracker.get_progress()
                print(progress)
                self.progress_bar["value"] = progress * 100
                self.update_idletasks()

    #pick and save video file
    def file_search(self):
        self.filename = filedialog.askopenfilename(initialdir='/home/moi/AGH/3rok/io/yolov4-deepsort-master/data/video', title="Wybierz obrazek",
                                                   filetypes=(("mpg", "*.mpg"), ("mp4", "*.mp4")))
        if self.filename == ():
            self.filename = ''
        if self.filename != '':
            directory = os.path.split(self.filename)[0]
            self.output_filename = os.path.join(directory, os.path.splitext(os.path.basename(self.filename))[0] + '_out.mp4')
            self.output_filename_log = os.path.join(directory,
                                                os.path.splitext(os.path.basename(self.filename))[0] + '_out_log.txt')
            self.label_file.grid(column=1, row=2)
            self.label_file.configure(text='Wybrano plik: ' + self.filename)


if __name__ == '__main__':
    root = Root()
    root.mainloop()
