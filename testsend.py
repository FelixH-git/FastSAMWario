import socket
import time
HOST = "127.0.0.1"
PORT = 5000

220,150,1300,960
data = "220,150,220,960,1300,960,1300,150"

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    print("Connected sucessfully")
    time.sleep(1)


    s.send(data.encode("cp1252"))
    print("Sent succesfully")
