from concurrent.futures import process
import socket
import struct
import websockets
import asyncio
import json
from TCP_Server.communication_parameters import REMOTE_UDP_ADDRESS
from communication_parameters import *

async def forward_message(message):
    url = 'ws://localhost:5000/echo'
    async with websockets.connect(url) as websocket:
        await websocket.send(message)


def xmit_loop(message):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(forward_message(message))

def processAnimationData(sock):
    data, addr = sock.recvfrom(4096)
    sock.sendto(data, ("localhost", 5003))
    #result = data.decode("utf-8")
    #asyncio.run(forward_message(result))
    #print("Data forwarded to websocket!")
if __name__ == '__main__':
    seq_num = -1

    udpClient_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udpClient_sock.bind((UDP_ADDRESS, UDP_PORT))

    while True:
        processAnimationData(udpClient_sock)
        # UDP socket's datagram-based, so we don't need the exact size, only the max size of the buffer (in bytes)
        # data, addr = udpClient_sock.recvfrom(4096)
        # seq_num_recv = struct.unpack_from("I", data, offset=0)[0]

        # # Reject datagrams arriving too late
        # if seq_num > 0 and abs(seq_num - seq_num_recv) > LATENESS_LIMIT:
        #     continue

        # result = struct.unpack_from("i", data, offset=4)

        # if seq_num_recv > seq_num:
        #     seq_num = seq_num_recv
