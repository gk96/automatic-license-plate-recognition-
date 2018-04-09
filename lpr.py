import tkinter as tk
import cv2
import pyocr
import pyocr.builders
from PIL import Image, ImageTk
import numpy as np
from pytesser3 import *
from ocr2 import mai

width, height = 700, 300
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

root = tk.Tk()
root.wm_title("License Plate Detector")
root.bind('<Escape>', lambda e: root.quit())
lmain = tk.Label(root)
l1=tk.Label(root)
lmain.pack(side="left",padx=10,pady=0)
l1.pack(side="left",padx=5,pady=5)
l11=tk.Label(root,text="Video Feed")
l11.config(font=("Courier",20))
l11.place(x=300,y=1)
l12=tk.Label(root,text="Plate")
l12.config(font=("Courier",20))
l12.place(x=700,y=1)
t=tk.Text(root,height=1,width=15)

car_cascade = cv2.CascadeClassifier('license.xml')

def show_frame():
    _, frame = cap.read()
    cv2.rectangle(frame, (100, 470), (500, 400), (255, 0, 0), 3)
    ro = frame[400:470, 100:500]
    gray = cv2.cvtColor(ro, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.equalizeHist(gray)

    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    for (x, y, w, h) in cars:


        cv2.rectangle(ro, (x, y), (x + w, y + h), (0, 0, 255), 2)

        img3=ro[y:y+h,x:x+w]
        con=np.array([[[x,y]],[[x+w,y]],[[x+w,y+h]],[[x,y+h]]])
        img4=cv2.medianBlur(img3,3)
        img4=cv2.threshold(img4,127,255,cv2.THRESH_BINARY)[1]
        img4 = cv2.medianBlur(img4, 3)
        img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('l3.jpg',img4)
        imh=Image.fromarray(img4)
        tools = pyocr.get_available_tools()[0]

        str = tools.image_to_string(imh,builder=pyocr.builders.DigitBuilder())


        t.delete("1.0",tk.END)

        t.insert(tk.END,str)
        t.pack(side="left",padx=5,pady=5)
        img4tk=ImageTk.PhotoImage(image=imh)
        l1.img4tk=img4tk
        l1.configure(image=img4tk)

    img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk

    lmain.configure(image=imgtk)

    lmain.after(10, show_frame)

show_frame()
root.mainloop()