import cv2
import dxcam
import torch
from fastsam import FastSAM, FastSAMPrompt
import time
import socket
import regex as re
import numpy as np
from tkinter import ttk
model = FastSAM('FastSAM-s.pt')

left, top = 220, 150
right, bottom = 1300,  960

region = (left, top, right, bottom)
print(torch.cuda.is_available(), "Torch avaliable?")
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def format(data: str):
    num = list(map(int, re.findall(r'\d+', data)))
    cleaned_data = [num[i:i+2] for i in range(0, len(num), 2)]
    return cleaned_data 

def test(screen, write_to_file=False):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = "127.0.0.1"
    port = 5000
    #s.connect((host,port))
    camera = dxcam.create(output_idx=screen)
    while(True):
        frame = camera.grab(region=region)
        if frame is None: continue
        
        start = time.perf_counter()
        
        everything_results = model(
        source=frame,
        conf=0.7,
        )
        
        
        prompt_process = FastSAMPrompt(frame, everything_results, device=DEVICE)
        if everything_results == None:
            continue
        masks_xy = everything_results[0].masks.xy

        ann = prompt_process.text_prompt("Laptop")
        
        
        end = time.perf_counter()
        if len(ann) == 0:
            continue
        img = prompt_process.plot_to_result(annotations=ann)

        total_time = end - start
        fps = 1/total_time
        
        data_string = str([xy.astype(int).tolist() for xy in masks_xy]).replace(" ", "").replace("]],[[", "]->[")[2:-2].replace("[", "").replace("]", "")

        print(data_string)
     #   s.send(data_string.encode("cp1252"))

        cv2.putText(img, f'FPS {int(fps)}', (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        cv2.imshow("Frame", img)
        if(cv2.waitKey(20) == ord("q")):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    test(0)