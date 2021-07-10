
from tkinter import *
from tkinter import messagebox, filedialog
from os import system
import moviepy.editor
import pymysql
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')
root=Tk()
import os
import tensorflow as tf
import cv2
import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import load_model
from clean import downsample_mono, envelope
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel

classes = {0: 'basketball', 1: 'boxing',
           2: 'cricket', 3: 'formula1',
           4: 'kabaddi', 5: 'swimming',
           6: 'table_tennis', 7: 'weight_lifting'}

for audios in ['data/music', 'data/speech']:
    for sport_folder in classes:
        folder_path = os.path.join(audios, classes[sport_folder])
        if not os.path.exists(folder_path):

            messagebox.showinfo("path", "Audio Path" + folder_path, parent=root)

            os.makedirs(folder_path)


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def search_list(vide_search):
    video_similarity_dict = {'basketball': 0, 'boxing': 0,
                             'cricket': 0, 'formula1': 0, 'kabaddi': 0,
                             'swimming': 0, 'table_tennis': 0, 'weight_lifting': 0}

    # vide_search=input('ADD:')# taking input

    for key in video_similarity_dict:
        score = similar(vide_search, key)
        video_similarity_dict[key] = score

    video_folder = max(video_similarity_dict, key=video_similarity_dict.get)

    # print(video_folder)

    speech_folder = 'data/speech/' + video_folder
    music_folder = 'data/music/' + video_folder

    speech_list = os.listdir(speech_folder)
    music_list = os.listdir(music_folder)

    files_dict = {}

    for i in speech_list:
        files_dict[i] = speech_folder

    for i in music_list:
        files_dict[i] = music_folder

        # print(speech_list)
    # print(music_list)
    return files_dict


def auio_video_merge(vid_path, audio_path):
    video_clip = moviepy.editor.VideoFileClip(vid_path)
    audioclip = moviepy.editor.AudioFileClip(audio_path)
    new_vid_path = vid_path.split('.', 1)[0] + '.mp4'
    # saving the clip
    videoclip = video_clip.set_audio(audioclip)
    videoclip.write_videofile(new_vid_path)
    print(new_vid_path)  # Video path
    # cam.release()
    # cv2.destroyAllWindows()
    videoclip.close()
    os.remove(vid_path)
    os.remove(audio_path)


    return new_vid_path


def spliting(filename):
    model_fn = 'models/lstm.h5'
    pred_fn = 'y_pred'
    dt = 1.0
    sr = 16000
    threshold = 20
    audio__class = ['music', 'speech']

    model = load_model(model_fn,
                       custom_objects={'STFT': STFT,
                                       'Magnitude': Magnitude,
                                       'ApplyFilterbank': ApplyFilterbank,
                                       'MagnitudeToDecibel': MagnitudeToDecibel})

    videoename = os.path.basename(filename)
    video = moviepy.editor.VideoFileClip(filename)  # Entering the videofile
    audio = video.audio
    audio_path = 'data/' + videoename.rsplit('.', 1)[0] + ".wav"
    audio.write_audiofile(audio_path)
    video.close()
    rate, wav = downsample_mono(audio_path, sr)
    mask, env = envelope(wav, rate, threshold=threshold)
    clean_wav = wav[mask]
    step = int(sr * dt)
    batch = []
    for i in range(0, clean_wav.shape[0], step):
        sample = clean_wav[i:i + step]
        sample = sample.reshape(-1, 1)
        if sample.shape[0] < step:
            tmp = np.zeros(shape=(step, 1), dtype=np.float32)
            tmp[:sample.shape[0], :] = sample.flatten().reshape(-1, 1)
            sample = tmp
        batch.append(sample)
    X_batch = np.array(batch, dtype=np.float32)
    y_pred = model.predict(X_batch)

    list = [[np.round(float(i), 2) for i in nested] for nested in y_pred]
    # print(list)

    y_mean = np.mean(list, axis=0)
    # print(y_mean)
    y_mean = [round(num) for num in y_mean]
    # print(y_mean)
    if y_mean[0] > y_mean[1]:
        y_pred = 0
    if y_mean[1] > y_mean[0]:
        y_pred = 1

    pred_audio_class = audio__class[y_pred]

    messagebox.showinfo("Info", "Audio of the video is "+pred_audio_class+' type', parent=root)

    model_path = 'models'
    # load json and create model
    json_file = open(os.path.join(model_path, 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(os.path.join(model_path, 'model.h5'))
    messagebox.showinfo("Model", "Loading Model From Disk", parent=root)

    model.compile(tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    classes = {0: 'basketball', 1: 'boxing',
               2: 'cricket', 3: 'formula1',
               4: 'kabaddi', 5: 'swimming',
               6: 'table_tennis', 7: 'weight_lifting'}

    cam = cv2.VideoCapture(filename)
    total_fps = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    prediction_counts = {'basketball': 0, 'boxing': 0,
                         'cricket': 0, 'formula1': 0, 'kabaddi': 0,
                         'swimming': 0, 'table_tennis': 0, 'weight_lifting': 0}
    # frame
    currentframe = 0
    messagebox.showinfo("Video", "Start video prediction", parent=root)

    pbar = tqdm(total=total_fps)
    while (True):
        # reading from frame
        ret, frame = cam.read()

        if ret:
            pred_img = cv2.resize(frame, (224, 224))
            pred_img = np.expand_dims(pred_img, axis=0)

            prediction = model.predict(pred_img)
            maxindex = int(np.argmax(prediction))
            sport = classes[maxindex]
            prediction_counts[sport] = prediction_counts[sport] + 1
            pbar.update(1)

            currentframe += 1
        else:
            cam.release()
            cv2.destroyAllWindows()
            # currentframe-=1
            break

    max_pred_sport = max(prediction_counts, key=prediction_counts.get)

    pred_ = (prediction_counts[max_pred_sport] / total_fps) * 100

    messagebox.showinfo("Prediction percentage", "Prediction percentage"+str(pred_) + '%', parent=root)

    folder_path = "data/" + pred_audio_class + '/' + max_pred_sport + '/'
    if not os.path.exists(folder_path):

        messagebox.showinfo("Making Path", "Making Path for video storage" + folder_path, parent=root)
        os.makedirs(folder_path)

    vid_path = folder_path + max_pred_sport + '_' + videoename.rsplit('.', 1)[0] + '.avi'


    messagebox.showinfo("video save location", "Writing Video in " + vid_path, parent=root)

    cam = cv2.VideoCapture(filename)
    fps = cam.get(cv2.CAP_PROP_FPS)
    fourcc = 'mp4v'
    w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    pbar = tqdm(total=total_fps)

    while (cam.isOpened()):
        ret, frame = cam.read()
        if ret == True:
            image = cv2.putText(frame, max_pred_sport, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 2, cv2.LINE_AA)
            vid_writer.write(image)
            pbar.update(1)
        else:
            break
    messagebox.showinfo("Process complete", "Video writing process complete", parent=root)

    cam.release()
    cv2.destroyAllWindows()
    # currentframe-=1
    return total_fps, vid_path, audio_path, max_pred_sport, pred_, pred_audio_class


def video_show(lb, all_files_list, files_dict):
    text = lb.get(ANCHOR)
    video_file_path = ""
    if bool(all_files_list):
        if text in all_files_list.keys():
            path = all_files_list[text]
            video_file_path = path + '/' + text

    if bool(files_dict):
        if text in files_dict.keys():
            path = files_dict[text]
            video_file_path = path + '/' + text


    messagebox.showinfo("Video path", "Playing video location at"+video_file_path, parent=root)

    cap = cv2.VideoCapture(video_file_path)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            img = cv2.resize(frame, (960, 540))
            cv2.imshow('Requested Video', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
        else:
            cap.release()
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()


class Main_App:

    def __init__(self, root):
        self.root = root
        self.root.geometry("500x500")
        self.root.resizable(False, False)
        self.root.configure(background="lightgrey")
        self.root.title("Main_App")
        btn_bar = Frame(root, width=500, highlightbackground='white', height='40').place(x='0', y='5')
        Frame.txt_name = Entry(root)
        Frame.txt_name.place(x=290, y=15)
        btn_search = Button(root, text='search', width=10, bg='#078fc9', font=('goudy old style', 10, 'bold'),
                            fg='white', command=self.search).place(x=415, y=12)
        btn_upload = Button(root, text='upload', width=10, bg='#078fc9', font=('goudy old style', 10, 'bold'),
                            fg='white', command=self.upload).place(x=200, y=12)
        btn_logout = Button(root, text='Logout', width=10, bg='#078fc9', font=('goudy old style', 10, 'bold'),
                            fg='white', command=self.logout).place(x=5, y=12)
        video_box = Frame(root, width=400, highlightbackground='white', height='300').place(x='50', y='100')

    #to search the user required category
    def search(self):
        #   messagebox.showinfo("search", "we are searching here", parent=self.root)
        Frame.txt_name.get()
        audio_lists = ['speech', 'music']
        files_dict = {}
        if not Frame.txt_name.get() in audio_lists:
            # input_string=Your input string that will comes from search box
            files_dict = search_list(Frame.txt_name.get())
            list = []
            for key in files_dict:
                list.append(key)
        #     print(list)
        # If user search speech or music:
        all_files_list = {}
        if Frame.txt_name.get() in audio_lists:
            paths = 'data/' + Frame.txt_name.get()
            folders = os.listdir(paths)
            for folder in folders:
                list_folder = os.listdir(paths + '/' + folder)
                for files in list_folder:
                    all_files_list[files] = paths + '/' + folder
            list = []
            for key in all_files_list:
                list.append(key)
        #   print(list)

        video_box = Frame(root, width=400, highlightbackground='white', height='300').place(x='50', y='100')

        lb = Listbox(
            video_box,
            width=47,
            height=14,
            font=('Times', 12),
            bd=0,
            fg='#464646',
            highlightthickness=0,
            selectbackground='#a6a6a6',
            activestyle="none")
        lb.place(x="58",y="112")
        # list = ['give', 'return'] # search result
        task_list = list  # here u will add list of video file
        for item in task_list:
            lb.insert(END, item)

        sb = Scrollbar(video_box,orient=VERTICAL)
        sb.place(x='450',y='112')

        lb.config(yscrollcommand=sb.set)
        sb.config(command=lb.yview)


        addTask_btn = Button(root, text='Play Video', width=10, bg='#078fc9', font=('goudy old style', 10, 'bold'),fg='white', command=lambda: video_show(lb, all_files_list, files_dict)).place(x="210",y="420")


    def upload(self):
        filename = filedialog.askopenfilename(initialdir="/", title="select a file",
                                              filetype=(("mp4", "*.mp4"), ("All Files", "*.*")))
        if (filename != ""):
            total_fps, vid_path, audio_path, max_pred_sport, pred_, pred_audio_class = spliting(filename)

            video_file_path = auio_video_merge(vid_path, audio_path)

            messagebox.showinfo("File Location", "File Save to " + pred_audio_class+" belongs to "+video_file_path+"sport", parent=root)

            no_f = str(total_fps)
            videoename = os.path.basename(filename)
            try:
                connection = pymysql.connect(host="localhost", user="root", password="", database="db_connectivity")
                cursor = connection.cursor()
                cursor.execute("insert into video_db (video) values (%s)", (filename))
                connection.commit()
                connection.close()
                messagebox.showinfo("Success",
                                    "Successfuly upload\n video splite into " + no_f + " \nClassified Video saved in \t" + video_file_path,
                                    parent=self.root)
                # system('Main_App.py')
            except Exception as es:
                messagebox.showerror("Error", f"Error due to:{str(es)}", parent=self.root)

    def logout(self):
        root.destroy()
        system('User_Login.py')

obj = Main_App(root)
root.mainloop()
