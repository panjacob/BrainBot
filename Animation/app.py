import json
import random
import struct
import time
import os
import socket

from flask import Flask, render_template
from flask_sock import Sock


app = Flask(__name__)
sock = Sock(app)


@app.route('/')
def index():
    # return "Hello world"
    return render_template('index.html'), 200


@sock.route('/echo')
def echo(sock):
    while True:
        data = sock.receive()
        #data = ws.receive()
        label = struct.unpack("i", data)
        time.sleep(2)
        left = False
        forward = (label == -1)
        data = {'left': left, "forward": forward}
        context = json.dumps(data)
        sock.send(context)


if __name__ == '__main__':
    app.run(host="localhost", port=5500)
