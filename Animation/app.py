import json
import random
import time

from flask import Flask, render_template
from flask_sock import Sock

app = Flask(__name__)
sock = Sock(app)


@app.route('/')
def index():
    return render_template('index.html')


@sock.route('/echo')
def echo(sock):
    while True:
        # data = sock.receive()
        time.sleep(2)
        left = random.choice([True, False])
        forward = random.choice([True, False])
        data = {'left': left, "forward": forward}
        context = json.dumps(data)
        sock.send(context)
