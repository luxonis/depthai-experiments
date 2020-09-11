#!/usr/bin/env python3

import os
import subprocess
from pathlib import Path
import signal
from flask import Flask, jsonify, request

app = Flask(__name__)
proc = None

depthai_path = Path('../depthai/depthai_demo.py').resolve()
output_file = None
cmd = f"python3 {depthai_path}"
cmd_env = os.environ.copy()
cmd_env["PYTHONPATH"] = str(depthai_path.parent) + ':' + cmd_env["PYTHONPATH"]


@app.route("/run/", methods=['POST'])
def run():
    global proc, cmd_env, cmd, depthai_path, output_file
    experiment = request.json.get("experiment", None) if request.json is not None else None
    if experiment is None:
        args = request.json.get("args", "") if request.json is not None else ""
        cmd_string = cmd + " " + args
        workdir = depthai_path.parent
    else:
        experiment_path = Path(f'../../{experiment}/main.py').resolve()
        cmd_env["PYTHONPATH"] = str(experiment_path.parent) + ':' + cmd_env["PYTHONPATH"]
        cmd_string = f"python3 {experiment_path}"
        workdir = experiment_path.parent
    if proc is not None and proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    if output_file is not None:
        output_file.close()
    output_file = open('proc.log', 'w+')
    proc = subprocess.Popen(cmd_string, shell=True, preexec_fn=os.setsid, env=cmd_env, cwd=workdir, stdout=output_file)
    return jsonify(success=True)


@app.route("/kill/", methods=['POST'])
def kill():
    global proc
    if proc is not None and proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        return jsonify(success=True)
    else:
        return jsonify(success=False)


@app.route("/status/", methods=['GET'])
def status():
    global proc
    if proc is None:
        return jsonify(status="empty")
    elif proc.poll() is None:
        return jsonify(status="running")
    else:
        return jsonify(status="killed")


@app.route("/logs/", methods=['GET'])
def logs():
    global proc, output_file
    if output_file is None:
        return jsonify(logs=[])
    output_file.seek(0)
    return jsonify(logs=output_file.readlines()[:20])


app.run(host='0.0.0.0', port=8080)
