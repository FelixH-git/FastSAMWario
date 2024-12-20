import ctypes
import numpy as np
import cv2
from subprocess import Popen, PIPE

def initialize_varjo():

    varjo_lib = ctypes.CDLL(r"C:\Users\isbl0003\Documents\varjo-sdk\bin\VarjoLib.dll")

    avail = varjo_lib.varjo_IsAvailable()
    if not avail:
        raise Exception("Varjo Headset not available")

    session = varjo_lib.varjo_SessionInit()
    if not session:
        raise Exception("Failed to create Varjo session")

    return varjo_lib, session

def fetch_mr_camera_feed(varjo_lib, session):

    class VarjoCameraFrame(ctypes.Structure):
        fields = [
            ("width", ctypes.cint),
            ("height", ctypes.cint),
            ("format", ctypes.cint),
            ("buffer", ctypes.POINTER(ctypes.cuint8)),
        ]

    varjo_get_camera_frame = varjo_lib.varjo_GetCameraImage
    varjo_get_camera_frame.restype = ctypes.POINTER(VarjoCameraFrame)

    frame_ptr = varjo_get_camera_frame(session)
    if not frame_ptr:
        raise Exception("Failed to get frame")

    frame = frame_ptr.contents


    frame_data = np.ctypeslib.as_array(frame.buffer, shape=(frame.height, frame.width, 3))
    return frame_data


def main():
    varjo_lib, session = initialize_varjo()
    print(session)

    print(varjo_lib.varjo_RequestGazeCalibration(session))

    print(varjo_lib.varjo_GetDataStreamConfigCount(session))

    varjo_lib.varjo_SessionShutDown(session)



if __name__ == "__main__":
    main()