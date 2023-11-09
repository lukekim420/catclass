import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import sounddevice as sd


import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def get_all_files():
    train_folder = glob(input_path_box.get("1.0", "end-1c"))

    all_files = []
    for folder in train_folder:
        tmp = glob(folder + '/*')
        all_files += tmp
        
    print(all_files)
    return all_files

def ctft_to_peaks(carr : np.ndarray, threshold = 1, max_peak_num = 10):
    rising = np.zeros(shape=(max_peak_num, carr.shape[1]))
    falling = np.zeros(shape=(max_peak_num , carr.shape[1]))
    amplitude = np.zeros(shape=(max_peak_num , carr.shape[1]))

    diff=np.diff(np.sign(carr.T-threshold), axis = 1)

    for arr, num in [(rising, 2), (falling, -2)]:
        a,b = np.where(diff==num)
        for i in range(diff.shape[0]):
            temp=b[a==i]
            minlen = min(max_peak_num, len(temp))
            arr[:minlen,i]=temp[:minlen]
                
    return (rising, falling)

class Classify_manager:
    def __init__(self, file_path) -> None:
        #load file
        y, sr = librosa.load(file_path)

        #노이즈 제거
        S_full, phase = librosa.magphase(librosa.stft(y)) 
        S_filter = librosa.decompose.nn_filter(S_full,
                                        aggregate=np.median,
                                        metric='cosine',
                                        width=int(librosa.time_to_frames(2, sr=sr)))
        
        S_filter = np.minimum(S_full, S_filter)
        margin_v = 10
        power = 2
        mask_v = librosa.util.softmask(S_full - S_filter,
                                    margin_v * S_filter,
                                    power=power)

        self.S_foreground = mask_v * S_full
        self.S_full = S_full
        self.y, self.sr = y, sr

        #자르기
        rising, falling = ctft_to_peaks(self.S_foreground, threshold=0.1, max_peak_num=5)
        plt.figure(figsize=(15,8))

        #의미있는 부분만 잘라내기
        cnt_mat = (rising > 1e-5).sum(axis=0)
        thres = 5 - 0.5
        diff=np.diff(np.sign(cnt_mat-thres))
        cutpoint_rising = np.where(diff==2)[0]
        cutpoint_falling = np.where(diff==-2)[0]+1
        if(cnt_mat[0] > thres):
            cutpoint_rising = np.insert(cutpoint_rising,0,0)
        if(cnt_mat[-1] > thres):
            cutpoint_falling = np.append(cutpoint_falling, len(cnt_mat)-1)

        self.rising, self.falling = rising, falling

        self.cutpoint_falling, self.cutpoint_rising = cutpoint_falling,cutpoint_rising

        self.cut_count = len(self.cutpoint_falling)
            
    #노이즈 제거된 전체 스펙트럼 표시
    def displaytotalspectrum(self, fig:Figure):
        plot = fig.add_subplot(1,1,1)
        fig.colorbar(librosa.display.specshow(librosa.amplitude_to_db(self.S_foreground, ref=np.max),
                         y_axis='hz', x_axis='time', sr=self.sr,ax=plot), ax=plot)
        
    def displaycut(self, fig : Figure, index : int):
        a,b=self.cutpoint_rising[index], self.cutpoint_falling[index]
        plot1, plot2 = tuple(fig.subplots(1, 2))
        librosa.display.specshow(librosa.amplitude_to_db(self.S_foreground[:,a:b], ref=np.max),
                        y_axis='hz', x_axis='time', sr=self.sr, ax = plot1)

        plot2.scatter(np.arange(a, b,0.2).astype(int), self.rising[:,a:b].T.flatten(), s=10, marker='_')

    def playtotal(self):
        sd.play(self.y, samplerate=self.sr)

    def playcut(self, index:int):
        a,b=self.cutpoint_rising[index], self.cutpoint_falling[index]
        an, bn = librosa.frames_to_samples(a), librosa.frames_to_samples(b)
        sd.play(self.y[an:bn], samplerate=self.sr)

    def savecut(self, path : str, index : int):
        pass

# Create the main window
window = tk.Tk()
window.title("Cat Classifier")
window.geometry("750x750")

input_path_box = tk.Text(window, width=80, height=1)
input_path_box.insert("1.0", './catclass/audioset-processing-master/output/meow')
input_path_box.pack()
input_path_box.place(x=20, y=20)

all_files = []
index = 0
filecount = 0
fileindex = 0
currentmanager : Classify_manager = None

def select_path_btn_pressed():
    global all_files, filecount, fileindex, index, currentmanager

    all_files = get_all_files()
    filecount = len(all_files)
    fileindex = 0
    index = 0

    currentmanager = Classify_manager(all_files[fileindex])
    
    select_path_btn['state']='disabled'
    nextcut_button['state']='enabled'
    nextfile_button['state']='enabled'
    playcut_button['state']='enabled'
    playall_button['state']='enabled'

    set_index_label()
    set_canvas()

select_path_btn = ttk.Button(window, text="파일 가져오기", command=select_path_btn_pressed)
select_path_btn.place(x=600, y=20)

def set_index_label():
    index_label['text'] = f'{fileindex+1}/{filecount} 번째 파일. {index+1}/{0 if currentmanager == None else currentmanager.cut_count} 번째 조각'

def set_canvas():
    currentmanager.displaytotalspectrum(fig=fig1)
    currentmanager.displaycut(fig=fig2, index=index)
    canvas1.draw()
    canvas2.draw()

index_label = ttk.Label(window, text = '0/0번째 파일. 0/0번째 조각')
index_label.place(x=20, y = 80)

############
fig1 = Figure(figsize=(6, 2), dpi=100)
canvas1 = FigureCanvasTkAgg(fig1, master=window)
canvas1.get_tk_widget().pack()
canvas1.get_tk_widget().place(x=20, y=120)


fig2 = Figure(figsize=(6,3), dpi=100)
canvas2 = FigureCanvasTkAgg(fig2, master=window)
canvas2.get_tk_widget().pack()
canvas2.get_tk_widget().place(x=20, y=370)

def nextcut_button_pressed():
    global index
    index+=1
    if index == currentmanager.cut_count:
        nextcut_button['state']='disabled'
    set_canvas()
    set_index_label()
    

nextcut_button = ttk.Button(window, text="다음 조각", command=nextcut_button_pressed)
nextcut_button.pack()
nextcut_button['state']='disabled'
nextcut_button.place(x=650, y=400)

def nextfile_button_pressed():
    global fileindex, currentmanager, index
    fileindex+=1
    index = 0
    currentmanager = Classify_manager(all_files[fileindex])
    if fileindex == len(all_files):
        nextfile_button['state']='disabled'
    set_canvas()
    set_index_label()

nextfile_button = ttk.Button(window, text="다음 파일", command=nextfile_button_pressed)
nextfile_button.pack()
nextfile_button['state']='disabled'
nextfile_button.place(x=650, y=450)

playall_button = ttk.Button(window, text="전체 재생", command=lambda : currentmanager.playtotal())
playall_button.pack()
playall_button['state']='disabled'
playall_button.place(x=650, y=500)

playcut_button = ttk.Button(window, text="조각 재생", command=lambda : currentmanager.playcut(index))
playcut_button.pack()
playcut_button['state']='disabled'
playcut_button.place(x=650, y=550)

# Start the Tkinter event loop
window.mainloop()
