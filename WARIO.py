import cv2
import dxcam
import torch
from fastsam import FastSAM, FastSAMPrompt
import time
import socket
import regex as re
import numpy as np
from threading import Thread
from tkinter import Tk, Label, Button, filedialog, StringVar, PhotoImage
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk

class AIApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("W.A.R.I.O")
        self.root.geometry("800x600")
        icon_image = PhotoImage(file="wario1.png")
        self.root.iconphoto(False, icon_image)

        # Set theme using ttkbootstrap's Style
        style = ttk.Style("superhero")  # Modern theme

        # Variables
        self.ip = StringVar(value="127.0.0.1")
        self.port = StringVar(value="5000")
        self.model_path = StringVar(value="FastSAM-s.pt")
        self.connection_active = False
        self.mode = StringVar(value="everything")  # Mode toggle
        self.prompt_text = StringVar(value="")  # For text_prompt
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        # Frames
        top_frame = ttk.Frame(root, padding=10)
        top_frame.pack(fill=X)

        control_frame = ttk.Frame(root, padding=10)
        control_frame.pack(fill=X)

        self.output_frame = ttk.LabelFrame(root, text="Output Preview", padding=10)
        self.output_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # UI Components
        ttk.Label(top_frame, text="IP Address:", bootstyle="info").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(top_frame, textvariable=self.ip, width=15).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(top_frame, text="Port:", bootstyle="info").grid(row=0, column=2, padx=5, pady=5)
        ttk.Entry(top_frame, textvariable=self.port, width=8).grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(top_frame, text="Model File:", bootstyle="info").grid(row=1, column=0, padx=5, pady=5)
        ttk.Entry(top_frame, textvariable=self.model_path, width=40).grid(row=1, column=1, columnspan=2, padx=5, pady=5)
        ttk.Button(top_frame, text="Browse", command=self.browse_model, bootstyle=INFO).grid(row=1, column=3, padx=5, pady=5)


        # Mode Selector
        ttk.Label(top_frame, text="Mode:", bootstyle="info").grid(row=2, column=0, padx=5, pady=5)
        ttk.Radiobutton(top_frame, text="Everything Prompt", variable=self.mode, value="everything", bootstyle="primary").grid(row=2, column=1, padx=5, pady=5)
        ttk.Radiobutton(top_frame, text="Text Prompt", variable=self.mode, value="text", bootstyle="primary").grid(row=2, column=2, padx=5, pady=5)

        # Prompt Input for Text Mode
        ttk.Label(top_frame, text="Prompt Text:", bootstyle="info").grid(row=3, column=0, padx=5, pady=5)
        ttk.Entry(top_frame, textvariable=self.prompt_text, width=40).grid(row=3, column=1, columnspan=2, padx=5, pady=5)


        self.start_button = ttk.Button(control_frame, text="Start", command=self.start_connection, bootstyle=SUCCESS)
        self.start_button.pack(side=LEFT, padx=10, pady=10)

        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_connection, bootstyle=DANGER, state="disabled")
        self.stop_button.pack(side=LEFT, padx=10, pady=10)

        self.canvas = ttk.Label(self.output_frame, text="No output yet", anchor=CENTER, bootstyle="secondary")
        self.canvas.pack(fill=BOTH, expand=True)

        # Thread control
        self.capture_thread = None
        self.camera = None

    

    def browse_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.pt")])
        if file_path:
            self.model_path.set(file_path)

    def start_connection(self):
        if self.connection_active:
            return

        self.connection_active = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.capture_thread = Thread(target=self.run_camera, daemon=True)
        self.capture_thread.start()

    def stop_connection(self):
        self.connection_active = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")

    def run_camera(self):
        left, top, right, bottom = 220, 150, 1300, 960
        region = (left, top, right, bottom)

        # Load model
        try:
            model = FastSAM(self.model_path.get())
        except Exception as e:
            self.update_canvas(f"Error loading model: {e}")
            self.stop_connection()
            return

        # Setup socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((self.ip.get(), int(self.port.get())))
        except Exception as e:
            self.update_canvas(f"Error connecting to socket: {e}")
            self.stop_connection()
            return

        # Initialize camera
        self.camera = dxcam.create(output_idx=0)

        while self.connection_active:
            frame = self.camera.grab(region=region)
            if frame is None:
                continue

            start = time.perf_counter()

            everything_results = model(
                source=frame,
                conf=0.7,
            )

            if everything_results is None:
                continue

            prompt_process = FastSAMPrompt(frame, everything_results, device=self.device)
            masks_xy = everything_results[0].masks.xy

            if self.mode.get() == "everything":
                ann = prompt_process.everything_prompt()
                
            else:
                prompt_text = self.prompt_text.get()
                ann = prompt_process.text_prompt(prompt_text)
            if len(ann) == 0:
                continue

            img = prompt_process.plot_to_result(annotations=ann)

            # FPS Calculations
            end = time.perf_counter()
            total_time = end - start
            fps = 1 / total_time

            # Resize image to 512x512
            img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)

            # Display FPS on frame
            cv2.putText(img_resized, f'FPS {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # # Convert frame for Tkinter display
            # img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            # img_pil = Image.fromarray(img_rgb)
            # img_tk = ImageTk.PhotoImage(img_pil)

            # self.update_canvas(img_tk)

            # Send data to the server
            data_list = []
            for segmentation in masks_xy:
                seg = segmentation.astype(int).tolist()
                data_list.append(seg)
            
            data_string = str(data_list).replace(" ", "").replace("]],[[", "]->[")[2:-2].replace("[", "").replace("]", "")
            # print(data_string)
            s.send(data_string.encode("cp1252"))

            if cv2.waitKey(20) == ord("q"):
                break

        s.close()
        self.camera.stop()
        cv2.destroyAllWindows()

    def update_canvas(self, content):
        if isinstance(content, ImageTk.PhotoImage):
            self.canvas.config(image=content, text="")
            self.canvas.image = content
        else:
            self.canvas.config(image="", text=content)

if __name__ == "__main__":
    root = ttk.Window("W.A.R.I.O", themename="superhero")
    app = AIApplication(root)
    root.mainloop()
