import cv2
import dxcam
import torch
from fastsam import FastSAM, FastSAMPrompt
import time

model = FastSAM('FastSAM-s.pt')



left, top = (1920 - 640) // 2, (1080 - 640) // 2
right, bottom = left + 640, top + 640

region = (left, top, right, bottom)
print(torch.cuda.is_available(), "Torch avaliable?")
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
def test(screen):

    camera = dxcam.create(output_idx=screen)
    with open("test.txt", "w") as f:

        while(True):
            frame = camera.grab(region=region)
            if frame is None: continue
            
            start = time.perf_counter()
            
            everything_results = model(
            source=frame,
            device=DEVICE,
            retina_masks=True,
            imgsz=1024,
            conf=0.4,
            iou=0.9,
            )
            
            
            prompt_process = FastSAMPrompt(frame, everything_results, device=DEVICE)

            ann = prompt_process.everything_prompt()
            #print(ann)
            
            end = time.perf_counter()
            img = prompt_process.plot_to_result(annotations=ann)
            f.write(str(img))
            total_time = end - start
            fps = 1/total_time
            


            cv2.putText(img, f'FPS {int(fps)}', (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
            cv2.imshow("Frame", img)
            if(cv2.waitKey(20) == ord("q")):
                cv2.destroyAllWindows()
                break

if __name__ == "__main__":
    test(0)