from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import numpy as np
import json
import os
import time
from obspy import read
from Seismic import from_ms2data, VitInference
from Seismic import STALTA
import yaml
import httpx
import matplotlib.pyplot as plt

app = FastAPI()

# UPLOAD_DIR = "uploads"
# PROCESSED_DIR = "processed"
# os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs(os.path.join(UPLOAD_DIR, PROCESSED_DIR), exist_ok=True)
sta_lst = STALTA()

def plot(st, arrival):
    fig, ax = plt.subplots(1,1,figsize=(12,3))
    ax.axvline(x = arrival, color='red', label='Trig. On')

    tr_times = st[0].times()
    tr_data = st[0].data    

    # Plot seismogram
    plt.plot(tr_times, tr_data)
    random_name = f"plot_{np.random.randint(10000)}.png"
    plt.savefig(f"../MoonQuake/analyzer/static/images/{random_name}") 
    return "/images/{random_name}"



@app.post("/get-stalta-startpoint")
async def get_quake_startpoint(file: UploadFile = File(...)):
    
    SAVE_FILE_PATH = "tmp/"
    content = await file.read()

    with open(SAVE_FILE_PATH+file.filename, "wb") as f:
        f.write(content)

    st = read(SAVE_FILE_PATH+file.filename)
    arrival = sta_lst.get_start(st)
    
    return JSONResponse({"startPoint": arrival, "path": plot(st, arrival)})


@app.post("/get-vit-startpoint")
async def get_quake_startpoint(file: UploadFile = File(...)):
    
    SAVE_FILE_PATH = "tmp/"
    content = await file.read()

    with open(SAVE_FILE_PATH+file.filename, "wb") as f:
        f.write(content)

    vel, mode, st = from_ms2data(SAVE_FILE_PATH+file.filename)
    use_vit = VitInference(model_path="vit_seismic.pt",data_max_length=572427)
    arrival = use_vit(vel=vel,mode=mode)


    return JSONResponse({"startPoint": arrival, "path": plot(st, arrival)})