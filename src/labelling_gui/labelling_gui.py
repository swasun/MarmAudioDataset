# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>

# License: BSD (3-clause)

import tkinter as Tk
from tkinter import messagebox
import tkinter.filedialog as fd
import os
from PIL import Image, ImageTk
import librosa
import sounddevice as sd
import pandas as pd
import argparse
import numpy as np
from matplotlib import cm
import pathlib
import random
import time
import re
import sys
import math


def create_circle(x, y, r, canvas, label):
    x0 = x - r
    y0 = y - r
    x1 = x + r
    y1 = y + r
    return canvas.create_oval(x0, y0, x1, y1, fill='red', tags=label)

def in_circle(center_x, center_y, radius, x, y):
    dist = math.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)
    return dist <= radius

def center(win):
    """
    centers a tkinter window
    :param win: the main window or Toplevel window to center
    """
    win.update_idletasks()
    width = win.winfo_width()
    frm_width = win.winfo_rootx() - win.winfo_x()
    win_width = width + 2 * frm_width
    height = win.winfo_height()
    titlebar_height = win.winfo_rooty() - win.winfo_y()
    win_height = height + titlebar_height + frm_width
    x = win.winfo_screenwidth() // 2 - win_width // 2
    y = win.winfo_screenheight() // 2 - win_height // 2
    win.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    win.deiconify()

def load_spectrogram_as_img(file_path):
    array = np.flipud(np.load(file_path))
    array = (array - np.min(array))/np.ptp(array) # Normalize between 0 and 1
    img = Image.fromarray(np.uint8(cm.gist_earth(array)*255))
    return img

class ResizingCanvas(Tk.Canvas):

    def __init__(self, parent, root_data_path, root_saving_path, session_id, load_spectrograms, file_names, **kwargs):
        Tk.Canvas.__init__(self, parent, **kwargs)
        self.root_data_path = root_data_path
        self.root_saving_path = root_saving_path
        self.session_id = session_id
        self.file_names = file_names
        self.width = kwargs['width']
        self.height = kwargs['height']
        self.image_on_canvas = None
        self.i = 0
        self.bind("<Configure>", self.on_resize)
        self.load_spectrograms = load_spectrograms
        self.circles = list()
        self.circle_index = 0

    def on_resize(self, event):
        self.width = event.width
        self.height = event.height
        self.config(width=self.width, height=self.height)

        if self.load_spectrograms:
            img = load_spectrogram_as_img(os.path.join(self.root_data_path, self.session_id, 'spectrograms', f'{self.file_names[self.i]}.npy'))
        else:
            f = open(os.path.join(self.root_data_path, self.session_id, 'images', f'{self.file_names[self.i]}.png'), 'rb')
            img = Image.open(f)
            img.load()
            f.close()

        img = img.resize((self.width, self.height), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(img)
        label = Tk.Label(image=photo)
        label.image = photo

        def on_click(event):
            if len(self.circles) > 0:
                for (circle, width, height, center_x, center_y, index) in self.circles:
                    if in_circle(center_x, center_y, 5, event.x, event.y):
                        self.delete(f'circle_{str(index)}')
                        self.circles.remove((circle, width, height, center_x, center_y, index))
                        return
            circle = create_circle(event.x, event.y, 5, self, label=f'circle_{str(self.circle_index)}')
            self.circles.append((circle, self.width, self.height, event.x, event.y, self.circle_index))
            self.circle_index += 1

        self.bind("<Button-1>", on_click)
        if not self.image_on_canvas:
            self.image_on_canvas = self.create_image(0, 0, image=photo, anchor=Tk.NW)
        else:
            self.itemconfig(self.image_on_canvas, image=photo)
        img.close()
        img = None

class CustomText(Tk.Text):
    '''A text widget with a new method, HighlightPattern 

    example:

    text = CustomText()
    text.tag_configure("red",foreground="#ff0000")
    text.HighlightPattern("this should be red", "red")

    The HighlightPattern method is a simplified python 
    version of the tcl code at http://wiki.tcl.tk/3246
    '''
    def __init__(self, *args, **kwargs):
        Tk.Text.__init__(self, *args, **kwargs)

    def HighlightPattern(self, pattern, tag, start="1.0", end="end", regexp=True):
        '''Apply the given tag to all text that matches the given pattern'''

        start = self.index(start)
        end = self.index(end)
        self.mark_set("matchStart",start)
        self.mark_set("matchEnd",end)
        self.mark_set("searchLimit", end)
        self.config(state=Tk.DISABLED)

        count = Tk.IntVar()
        while True:
            index = self.search(pattern, "matchEnd","searchLimit",count=count, regexp=regexp)
            if index == "": break
            self.mark_set("matchStart", index)
            self.mark_set("matchEnd", "%s+%sc" % (index,count.get()))
            self.tag_add(tag, "matchStart","matchEnd")

class Application(Tk.Frame):

    def __init__(self, parent, root_data_path, root_saving_path, session_id, sr, load_spectrograms, vocalization_types_file_name, vocalization_groups_file_name,
        monkey_identities_file_name, relabel, relabelling_target, new_limits_mandatory):
        Tk.Frame.__init__(self, parent)
        self.parent = parent
        self.relabel = relabel
        self.parent.grid_rowconfigure(0, weight=1)
        self.parent.grid_columnconfigure(0, weight=1)
        self.i = 0
        self.current_session_iterations = 0
        self.saving_every = 5
        self.saving_counter = 0
        self.data = []
        self.parent.protocol('WM_DELETE_WINDOW', self.exit_command)
        self.sr = sr
        self.root_data_path = root_data_path
        self.root_saving_path = root_saving_path
        self.session_id = session_id
        self.previous_vocalization_labels = []
        self.previous_monkey_identities = []
        self.selected_vocalization_listbox_ids = []
        self.selected_identity_listbox_ids = []
        self.new_limits_mandatory = new_limits_mandatory

        menu_bar = Tk.Menu(self)
        shortcut_bar = Tk.Menu(menu_bar, tearoff=False)
        menu_bar.add_cascade(label="Help", underline=0, menu=shortcut_bar)
        shortcut_bar.add_command(label="Shortcut", underline=1,
                             command=self.do_help, accelerator="Ctrl+H")
        self.bind_all("<Control-h>", self.do_help_event)
        parent.config(menu=menu_bar)

        pathlib.Path(os.path.join(self.root_saving_path, self.session_id)).mkdir(parents=True, exist_ok=True)
        if os.sep in self.session_id:
            self.labels_file_path = os.path.join(self.root_saving_path, self.session_id, f'{self.session_id.split(os.sep)[-1]}_expert_labels.tsv')
        else:
            self.labels_file_path = os.path.join(self.root_saving_path, self.session_id, f'{self.session_id}_expert_labels.tsv')
        self.load_spectrograms = load_spectrograms

        if os.path.isfile(self.labels_file_path):
            self.labels = pd.read_csv(self.labels_file_path, sep='\t')
            self.labels = self.labels.loc[:, ~self.labels.columns.str.contains('^Unnamed')]
            if 'identity' not in self.labels.columns:
                self.labels['identity'] = ''
                self.labels.reset_index(drop=True).to_csv(self.labels_file_path, sep='\t')
            if 'group' not  in self.labels.columns:
                self.labels['group'] = ''
                self.labels.reset_index(drop=True).to_csv(self.labels_file_path, sep='\t')
            if 'onset' not in self.labels.columns:
                self.labels['onset'] = ''
                self.labels.reset_index(drop=True).to_csv(self.labels_file_path, sep='\t')
            if 'offset' not in self.labels.columns:
                self.labels['offset'] = ''
                self.labels.reset_index(drop=True).to_csv(self.labels_file_path, sep='\t')
            if 'relabeled' not in self.labels.columns:
                self.labels['relabeled'] = False
                self.labels.reset_index(drop=True).to_csv(self.labels_file_path, sep='\t')
            if relabel:
                if relabelling_target is not None:
                    if relabelling_target == 'no_identity':
                        self.labels = self.labels[self.labels.identity.isnull()]
                    elif relabelling_target.startswith('category_'):
                        self.labels = self.labels[self.labels.type == relabelling_target.split('_')[1]]
                    elif relabelling_target.startswith('group_'):
                        self.labels = self.labels[self.labels.group == relabelling_target.split('_')[1]]
                    elif relabelling_target == 'identity_':
                        self.labels = self.labels[self.labels.identity == relabelling_target.split('_')[1]]
                print(len(self.labels))
                self.labels = self.labels[self.labels.relabeled == False]
                print(len(self.labels))
                all_file_names = self.labels.id.tolist()
                self.file_names = all_file_names
                self.labelled_count = 0
                self.max_labelled_count = len(self.file_names)
            else:
                already_exist_file_names = self.labels['id'].tolist()
                all_file_names = [os.path.splitext(file_name)[0] for file_name in sorted(os.listdir(os.path.join(self.root_data_path, self.session_id, 'audios')))]
                self.file_names = list(set(all_file_names) - set(already_exist_file_names))
                self.labelled_count = len(list(set(already_exist_file_names)))
                self.max_labelled_count = len(list(set(all_file_names)))
            if len(self.file_names) == 0:
                messagebox.showinfo('Quit', 'Labelling done.')
                self.parent.destroy()
        else:
            self.file_names =  [os.path.splitext(file_name)[0] for file_name in sorted(os.listdir(os.path.join(self.root_data_path, self.session_id, 'audios')))]
            self.labels = None
            self.labelled_count = 0
            self.max_labelled_count = len(self.file_names)

        random.shuffle(self.file_names)

        vocalization_groups = []
        if vocalization_groups_file_name is not None and os.path.isfile(vocalization_groups_file_name):
            with open(vocalization_groups_file_name, 'r') as f:
                vocalization_groups = f.readlines()
                vocalization_groups = [item.strip() for item in vocalization_groups]
            self.vocalization_groups_to_listbox_id = {voc_group:index for index, voc_group in enumerate(vocalization_groups)}
            self.listbox_id_to_vocalization_groups = {index:voc_group for index, voc_group in enumerate(vocalization_groups)}

        vocalization_types = []
        if vocalization_types_file_name:
            with open(vocalization_types_file_name, 'r') as f:
                vocalization_types = f.readlines()
                vocalization_types = [item.strip() for item in vocalization_types]
            self.vocalization_types_to_listbox_id = {voc_type:index for index, voc_type in enumerate(vocalization_types)}
            self.listbox_id_to_vocalization_types = {index:voc_type for index, voc_type in enumerate(vocalization_types)}
        else:
            self.vocalization_types_to_listbox_id = None
            self.listbox_id_to_vocalization_types = None

        monkey_identities = []
        if monkey_identities_file_name is not None and os.path.isfile(monkey_identities_file_name):
            with open(monkey_identities_file_name, 'r') as f:
                monkey_identities = f.readlines()
                monkey_identities = [item.strip() for item in monkey_identities]
        self.monkey_identities_to_listbox_id = {identity:index for index, identity in enumerate(monkey_identities)}
        self.listbox_id_to_monkey_identities = {index:identity for index, identity in enumerate(monkey_identities)}

        # Canvas widget which contains the spectrogram img
        if self.load_spectrograms:
            ref_img = load_spectrogram_as_img(os.path.join(self.root_data_path, self.session_id, 'spectrograms', f'{self.file_names[self.i]}.npy'))
        else:
            ref_img = Image.open(os.path.join(self.root_data_path, self.session_id, 'images', f'{self.file_names[self.i]}.png'))
        width = ref_img.size[0]
        height = ref_img.size[1]
        self.canvas = ResizingCanvas(self.parent, root_data_path=self.root_data_path, root_saving_path=self.root_data_path, session_id=self.session_id,
            load_spectrograms=self.load_spectrograms, file_names=self.file_names,
            bg='white', width=width, height=height)
        self.canvas.grid(row=0, column=0, sticky='news')

        listbox_frame = Tk.PanedWindow(self.parent)
        listbox_frame.grid(column=1, row=0)

        self.progression_label = Tk.Label(listbox_frame, text='')
        self.progression_label.grid(row=0, column=0, sticky='news', pady=20)
        self.progression_label.config(text=f'Progression:\n{self.labelled_count}/{self.max_labelled_count}\n{self.file_names[self.i]}')

        next_listbox_frame_row = 1

        # Listbox with labelling groups
        if len(vocalization_groups) > 0:
            Tk.Label(listbox_frame, text='Groups').grid(row=next_listbox_frame_row, column=0, sticky='news')
            self.listbox_vocalization_groups = Tk.Listbox(listbox_frame, height=len(vocalization_groups))
            self.listbox_vocalization_groups.grid(row=next_listbox_frame_row+1, column=0, sticky='news')
            for item_index, item in enumerate(vocalization_groups):
                self.listbox_vocalization_groups.insert(item_index, item)
            next_listbox_frame_row += 2
        else:
            self.listbox_vocalization_groups = None

        # Listbox with labelling types
        if self.new_limits_mandatory == False:
            Tk.Label(listbox_frame, text='Categories').grid(row=next_listbox_frame_row, column=0, sticky='news')
            self.listbox_vocalization_types = Tk.Listbox(listbox_frame, height=len(vocalization_types), selectmode=Tk.MULTIPLE, exportselection=False)
            self.listbox_vocalization_types.grid(row=next_listbox_frame_row+1, column=0, sticky='news')
            for item_index, item in enumerate(vocalization_types):
                self.listbox_vocalization_types.insert(item_index, item)

        self.label_type = Tk.Label(listbox_frame, text='', font='Helvetica 13 bold')
        self.label_type.grid(row=next_listbox_frame_row+2, column=0, sticky='news')
        next_listbox_frame_row += 3

        # Listbox with monkey identities
        if len(monkey_identities) > 0:
            Tk.Label(listbox_frame, text='Identities').grid(row=next_listbox_frame_row+4, column=0, sticky='news')
            self.listbox_monkey_identities = Tk.Listbox(listbox_frame, height=len(monkey_identities), selectmode=Tk.MULTIPLE, exportselection=False)
            self.listbox_monkey_identities.grid(row=next_listbox_frame_row+5, column=0, sticky='news')
            if len(monkey_identities) > 0:
                for item_index, item in enumerate(monkey_identities):
                    self.listbox_monkey_identities.insert(item_index, item)
        else:
            self.listbox_monkey_identities = None

        button_frame = Tk.PanedWindow(self.parent)
        button_frame.grid(column=0, row=1)

        Tk.Button(button_frame, text='Next', command=self.next_command).grid(row=0, column=0, sticky='news')
        Tk.Button(button_frame, text='Play', command=self.play_command).grid(row=0, column=1, sticky='news')
        Tk.Button(button_frame, text='Exit', command=self.exit_command).grid(row=0, column=2, sticky='news')
        Tk.Label(button_frame, text='Other:').grid(row=0, column=3, sticky='news')
        self.edit_other = Tk.Entry(button_frame)
        self.edit_other.grid(row=0, column=4, sticky='news')
        Tk.Label(button_frame, text='Comment:').grid(row=0, column=5, sticky='news')
        self.edit_comment = Tk.Entry(button_frame)
        self.edit_comment.grid(row=0, column=6, sticky='news')

        self.load_and_display_spectrogram(self.i)

        def select_or_deselect(listbox, index, listbox_target):
            if listbox is None:
                return
            if index in listbox.curselection():
                listbox.selection_clear(index)
                if listbox_target == 'vocalization':
                    if index in self.selected_vocalization_listbox_ids:
                        self.selected_vocalization_listbox_ids.remove(index)
                elif listbox_target == 'identity':
                    if index in self.selected_identity_listbox_ids:
                        self.selected_identity_listbox_ids.remove(index)
            else:
                listbox.select_set(index)
                if listbox_target == 'vocalization':
                    self.selected_vocalization_listbox_ids.append(index)
                elif listbox_target == 'identity':
                    self.selected_identity_listbox_ids.append(index)

        def on_key_pressed(event):
            if event.char == ' ':
                self.play_command()
            elif event.char == '\r':
                self.previous_vocalization_labels = []
                self.previous_monkey_identities = []
                self.next_command()
            elif event.char == '\x1b':
                self.exit_command()
            elif event.char == '\x08':
                # Backspace
                self.previous_vocalization_labels = []
                self.previous_monkey_identities = []
                self.back_command()
            elif event.char == '&':
                self.previous_vocalization_labels = []
                select_or_deselect(self.listbox_vocalization_types, 0, listbox_target='vocalization')
            elif event.char == 'é':
                self.previous_vocalization_labels = []
                select_or_deselect(self.listbox_vocalization_types, 1, listbox_target='vocalization')
            elif event.char == '"':
                self.previous_vocalization_labels = []
                select_or_deselect(self.listbox_vocalization_types, 2, listbox_target='vocalization')
            elif event.char == '\'':
                self.previous_vocalization_labels = []
                select_or_deselect(self.listbox_vocalization_types, 3, listbox_target='vocalization')
            elif event.char == '(':
                self.previous_vocalization_labels = []
                select_or_deselect(self.listbox_vocalization_types, 4, listbox_target='vocalization')
            elif event.char == '-':
                self.previous_vocalization_labels = []
                select_or_deselect(self.listbox_vocalization_types, 5, listbox_target='vocalization')
            elif event.char == 'è':
                self.previous_vocalization_labels = []
                select_or_deselect(self.listbox_vocalization_types, 6, listbox_target='vocalization')
            elif event.char == '_':
                self.previous_vocalization_labels = []
                select_or_deselect(self.listbox_vocalization_types, 7, listbox_target='vocalization')
            elif event.char == 'ç':
                self.previous_vocalization_labels = []
                select_or_deselect(self.listbox_vocalization_types, 8, listbox_target='vocalization')
            elif event.char == 'à':
                self.previous_vocalization_labels = []
                select_or_deselect(self.listbox_vocalization_types, 9, listbox_target='vocalization')
            elif event.char == ')':
                self.previous_vocalization_labels = []
                select_or_deselect(self.listbox_vocalization_types, 10, listbox_target='vocalization')
            elif event.char == '=':
                self.previous_vocalization_labels = []
                select_or_deselect(self.listbox_vocalization_types, 11, listbox_target='vocalization')

            elif event.char == 'a':
                self.previous_monkey_identities = []
                select_or_deselect(self.listbox_monkey_identities, 0, listbox_target='identity')
            elif event.char == 'z':
                self.previous_monkey_identities = []
                select_or_deselect(self.listbox_monkey_identities, 1, listbox_target='identity')
            elif event.char == 'e':
                self.previous_monkey_identities = []
                select_or_deselect(self.listbox_monkey_identities, 2, listbox_target='identity')
            elif event.char == 'r':
                self.previous_monkey_identities = []
                select_or_deselect(self.listbox_monkey_identities, 3, listbox_target='identity')
            elif event.char == 't':
                self.previous_monkey_identities = []
                select_or_deselect(self.listbox_monkey_identities, 4, listbox_target='identity')
            elif event.char == 'y':
                self.previous_monkey_identities = []
                select_or_deselect(self.listbox_monkey_identities, 5, listbox_target='identity')

            elif event.char == 'g':
                select_or_deselect(self.listbox_vocalization_groups, 0, listbox_target='group')
            elif event.char == 'h':
                select_or_deselect(self.listbox_vocalization_groups, 1, listbox_target='group')
            elif event.char == 'j':
                select_or_deselect(self.listbox_vocalization_groups, 2, listbox_target='group')
            elif event.char == 'k':
                select_or_deselect(self.listbox_vocalization_groups, 3, listbox_target='group')
            elif event.char == 'l':
                select_or_deselect(self.listbox_vocalization_groups, 4, listbox_target='group')
            elif event.char == 'm':
                select_or_deselect(self.listbox_vocalization_groups, 5, listbox_target='group')

        parent.bind('<KeyPress>', on_key_pressed)

        center(self.parent)

    def do_help_event(self, event):
        self.do_help()

    def do_help(self):
        help_window = Tk.Toplevel()
        help_window.title("Shortcuts")
        help_window.iconbitmap('icon.ico')
        about = '''<Space> Play the vocalization
        <Enter> Pass to the next vocalization
        <Backspace> Go back to the previous vocalization
        <Escape> Exit the program\n
        Vocalization groups: g, h, j, k, ...
        Vocalization types: &, é, ", ...
        Monkey identities: a, z, e, r, ...'''
        about = re.sub("\n\s*", "\n", about) # remove leading whitespace from each line
        t=CustomText(help_window, wrap="word", width=100, height=10, borderwidth=0)
        t.tag_configure("blue", foreground="blue")
        t.pack(sid="top",fill="both",expand=True)
        t.insert("1.0", about)
        t.HighlightPattern("^.*? - ", "blue")
        Tk.Button(help_window, text='OK', command=help_window.destroy).pack()
        center(help_window)
        help_window.bind('<Escape>', lambda e: help_window.destroy())

    def load_and_display_spectrogram(self, i):
        self.canvas.delete(Tk.ALL)
        self.canvas.i = i
        self.progression_label.config(text=f'Progression:\n{self.labelled_count}/{self.max_labelled_count}\n{self.file_names[self.i]}')

        if self.load_spectrograms:
            img = load_spectrogram_as_img(os.path.join(self.root_data_path, self.session_id, 'spectrograms', f'{self.file_names[self.i]}.npy'))
        else:
            f = open(os.path.join(self.root_data_path, self.session_id, 'images', f'{self.file_names[i]}.png'), 'rb')
            img = Image.open(f)
            img.load()
            f.close()

        img = img.resize((self.canvas.width, self.canvas.height), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(img)
        label = Tk.Label(image=photo)
        label.image = photo
        item = self.canvas.create_image(0, 0, image=photo, anchor=Tk.NW)
        img.close()
        img = None

        if self.labels is not None and self.new_limits_mandatory == False:
            current_label_row = self.labels[self.labels.id == self.file_names[self.i]]
            if len(current_label_row) > 0:
                current_label = current_label_row['type'].values[0]
                self.label_type.config(text=f'Current: {current_label}')
                if '-' in current_label:
                    current_labels = current_label.split('-')
                else:
                    current_labels = [current_label]
                self.previous_vocalization_labels = current_labels
                for current_label in current_labels:
                    self.listbox_vocalization_types.select_set(self.vocalization_types_to_listbox_id[current_label])

                if not current_label_row['identity'].isnull().values[0]:
                    current_label_identity = current_label_row['identity'].values[0]
                    self.label_type['text'] = f'Previous: {current_label_identity}'
                    if '-' in current_label_identity:
                        current_label_identities = current_label_identity.split('-')
                    else:
                        current_label_identities = [current_label_identity]
                    self.previous_monkey_identities = current_label_identities
                    for current_label_identity in current_label_identities:
                        self.listbox_monkey_identities.select_set(self.monkey_identities_to_listbox_id[current_label_identity])

        self.audio = librosa.load(os.path.join(self.root_data_path, self.session_id, 'audios', f'{self.file_names[self.i]}.wav'), sr=self.sr)[0]
        self.begin_time = time.time()

    def next_command(self):
        selected = 0
        if self.new_limits_mandatory == False and len(self.listbox_vocalization_types.curselection()) > 0:
            selected += 1
        if self.listbox_vocalization_groups is not None and len(self.listbox_vocalization_groups.curselection()) > 0:
            selected += 1
        if self.listbox_monkey_identities is not None and len(self.listbox_monkey_identities.curselection()) > 0:
            selected += 1
        if selected == 0 and self.new_limits_mandatory == False:
            messagebox.showinfo('Warning', 'No item selected')
            return
        if len(self.canvas.circles) > 2:
            messagebox.showinfo('Warning', 'Only one onset and one offset can be specified')
            return
        if self.new_limits_mandatory and len(self.canvas.circles) != 2:
            messagebox.showinfo('Warning', 'Both onset and offset must be specified')
            return

        self.labelling_time = time.time() - self.begin_time
        edit_comment = self.edit_comment.get()

        if self.new_limits_mandatory:
            item = ''
        else:
            if len(self.previous_vocalization_labels) > 0:
                item = ''
                for current_item in self.previous_vocalization_labels:
                    if item == '':
                        item = current_item
                    else:
                        item += '-' + current_item
            else:
                if 'Other' in [self.listbox_vocalization_types.get(index) for index in self.listbox_vocalization_types.curselection()]:
                    edit_text = self.edit_other.get()
                    if edit_text != '':
                        item = edit_text
                    else:
                        item = 'Other'
                elif len(self.selected_vocalization_listbox_ids) > 0:
                    item = ''
                    for current_selection in self.selected_vocalization_listbox_ids:
                        current_item = self.listbox_id_to_vocalization_types[current_selection]
                        if item == '':
                            item = current_item
                        else:
                            item += '-' + current_item
                else:
                    item = ''
                    for current_selection in self.listbox_vocalization_types.curselection():
                        current_item = self.listbox_id_to_vocalization_types[current_selection]
                        if item == '':
                            item = current_item
                        else:
                            item += '-' + current_item

        monkey_identity = ''
        for current_selection in self.selected_identity_listbox_ids:
            current_item = self.listbox_id_to_monkey_identities[current_selection]
            if monkey_identity == '':
                monkey_identity = current_item
            else:
                monkey_identity += '-' + current_item

        current_vocalization_group = ''
        if self.listbox_vocalization_groups is not None:
            current_vocalization_groups = [self.listbox_vocalization_groups.get(index) for index in self.listbox_vocalization_groups.curselection()]
            if len(current_vocalization_groups) > 0:
                current_vocalization_group = current_vocalization_groups[0]

        self.previous_vocalization_labels = []
        self.previous_monkey_identities = []

        if self.relabel:
            labels = pd.read_csv(self.labels_file_path, sep='\t')
            labels = labels.loc[:, ~labels.columns.str.contains('^Unnamed')]
            labels = labels[~(labels.id == self.file_names[self.i])]
            labels.reset_index(drop=True).to_csv(self.labels_file_path, sep='\t')

        if len(self.canvas.circles) == 1:
            onset_circle = self.canvas.circles[0]
            offset_circle = None
        elif len(self.canvas.circles) == 2:
            if self.canvas.circles[0][3] <= self.canvas.circles[1][3]:
                onset_circle = self.canvas.circles[0]
                offset_circle = self.canvas.circles[1]
            else:
                onset_circle = self.canvas.circles[1]
                offset_circle = self.canvas.circles[0]
        else:
            onset_circle = None
            offset_circle = None
        duration = len(self.audio) / self.sr
        if onset_circle:
            onset = np.linspace(0.0, duration, onset_circle[1])[onset_circle[3]]
        else:
            onset = ''
        if offset_circle:
            offset = np.linspace(0.0, duration, offset_circle[1])[offset_circle[3]]
        else:
            offset = ''

        relabeled = True if self.relabel else False

        self.data.append((self.file_names[self.i], item, edit_comment, duration, self.labelling_time, monkey_identity, current_vocalization_group, onset, offset, relabeled))

        self.selected_vocalization_listbox_ids = []
        self.selected_identity_listbox_ids = []

        self.i += 1
        self.labelled_count += 1
        self.current_session_iterations += 1
        if self.i >= len(self.file_names):
            messagebox.showinfo('Quit', 'Labelling done.')
            self.save_labels()
            self.parent.destroy()
            sys.exit(0)
        elif self.saving_counter+1 == self.saving_every:
            self.save_labels()
            self.data = []
            self.saving_counter = 0
        else:
            self.saving_counter += 1

        if self.new_limits_mandatory == False:
            self.listbox_vocalization_types.selection_clear(0, Tk.END)
            if self.listbox_monkey_identities:
                self.listbox_monkey_identities.selection_clear(0, Tk.END)
            if self.listbox_vocalization_groups:
                self.listbox_vocalization_groups.selection_clear(0, Tk.END)
        self.edit_other.delete(0, Tk.END)
        self.label_type.configure(text='')
        del self.canvas.circles
        self.canvas.circles = list()
        self.canvas.circle_index = 0

        self.load_and_display_spectrogram(self.i)

    def play_command(self):
        sd.play(self.audio, self.sr, blocking=False)

    def exit_command(self):
        if messagebox.askokcancel('Quit', 'Do you want to quit?'):
            self.save_labels()
            self.parent.destroy()

    def back_command(self):
        if self.current_session_iterations == 0:
            return
    
        self.labelling_time = time.time() - self.begin_time

        if len(self.data) > 0:
            # Unsaved labelled data are still in the buffer. Removing the last item.
            self.data.pop()
            self.saving_counter -= 1
        elif self.current_session_iterations > 0:
            # No labelled data in the buffer. Loading the labels's file and removing the last line.
            labels = pd.read_csv(self.labels_file_path, sep='\t')
            labels = labels.loc[:, ~labels.columns.str.contains('^Unnamed')]
            labels.drop(labels.tail(1).index, inplace=True) # drop last row
            labels.reset_index(drop=True).to_csv(self.labels_file_path, sep='\t')

        self.listbox_vocalization_types.selection_clear(0, Tk.END)
        self.edit_other.delete(0, Tk.END)
        self.i -= 1
        self.labelled_count -= 1
        self.load_and_display_spectrogram(self.i)

    def save_labels(self):
        if len(self.data) > 0:
            print('Saving the labels...')
            columns = ['id', 'type', 'comment', 'duration', 'labelling_time', 'identity', 'group', 'onset', 'offset', 'relabeled']
            if os.path.isfile(self.labels_file_path):
                df = pd.DataFrame(self.data, columns=columns).reset_index(drop=True)
                df.to_csv(self.labels_file_path, sep='\t', mode='a', header=False)
            else:
                df = pd.DataFrame(self.data, columns=columns).reset_index(drop=True)
                df.to_csv(self.labels_file_path, sep='\t', columns=columns)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_data_path', nargs='?', type=str, default=None)
    parser.add_argument('--root_saving_path', nargs='?', type=str, default=None)
    parser.add_argument('--session_id', nargs='?', type=str, default=None)
    parser.add_argument('--vocalization_types_file_name', nargs='?', type=str, default=None)
    parser.add_argument('--sr', nargs='?', type=int, default=96000)
    parser.add_argument('--load_spectrograms', action='store_true', default=False, help='Looking for a folder containing the spectrograms instead of the images.')
    parser.add_argument('--vocalization_groups_file_name', nargs='?', type=str, default=None)
    parser.add_argument('--monkey_identities_file_name', nargs='?', type=str, default=None)
    parser.add_argument('--relabel', action='store_true', default=False)
    parser.add_argument('--relabelling_target', nargs='?', type=str, default=None)
    parser.add_argument('--new_limits_mandatory', action='store_true', default=False)

    args = parser.parse_args()

    window = Tk.Tk()
    window.geometry('500x500')
    window.title('Vocalization labelling')
    window.grid_rowconfigure(0, weight=1)
    window.grid_columnconfigure(0, weight=1)
    try:
        window.iconbitmap('icon.ico')
    except:
        print(f'[ERROR] icon.ico not found')
    #window.tk.call('wm','iconphoto', window._w, Tk.Image("photo", file="icon.png"))

    if args.root_data_path is None or args.root_saving_path is None or args.session_id is None:
        specified_dir_path = fd.askdirectory(parent=window, initialdir=os.getcwd(), title='Please select the session directory')
        split_paths = os.path.split(specified_dir_path)
        args.root_data_path = split_paths[0]
        args.root_saving_path = split_paths[0]
        args.session_id = split_paths[1]

    Application(window, root_data_path=args.root_data_path, root_saving_path=args.root_saving_path, session_id=args.session_id, sr=args.sr, load_spectrograms=args.load_spectrograms,
        vocalization_types_file_name=args.vocalization_types_file_name, monkey_identities_file_name=args.monkey_identities_file_name,
        vocalization_groups_file_name=args.vocalization_groups_file_name,
        relabel=args.relabel, relabelling_target=args.relabelling_target, new_limits_mandatory=args.new_limits_mandatory).parent.mainloop()

if __name__ == "__main__":
    main()
