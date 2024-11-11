import cv2
import dxcam
import torch
from fastsam import FastSAM, FastSAMPrompt
import time
import socket
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
def test(screen, write_to_file=False):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = "127.0.0.1"
    port = 27015
    #s.connect((host,port))
    camera = dxcam.create(output_idx=screen)
    while(True):
        frame = camera.grab(region=region)
        if frame is None: continue
        
        start = time.perf_counter()
        
        everything_results = model(
        source=frame,
        device=DEVICE,
        retina_masks=True,
        imgsz=1080,
        conf=0.7,
        iou=0.9,
        )
        
        
        prompt_process = FastSAMPrompt(frame, everything_results, device=DEVICE)

        ann = prompt_process.everything_prompt()
        #print(ann)
        
        end = time.perf_counter()
        if len(ann) == 0:
            continue
        img = prompt_process.plot_to_result(annotations=ann)

        total_time = end - start
        fps = 1/total_time
        
       # s.send(str(img).encode())

        cv2.putText(img, f'FPS {int(fps)}', (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        cv2.imshow("Frame", img)
        if(cv2.waitKey(100) == ord("q")):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    test(0)