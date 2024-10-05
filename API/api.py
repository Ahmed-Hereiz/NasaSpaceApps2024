from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import json
import os
import time
from obspy import read
from Seismic import STALTA
import yaml
import httpx

app = FastAPI()

# UPLOAD_DIR = "uploads"
# PROCESSED_DIR = "processed"
# os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs(os.path.join(UPLOAD_DIR, PROCESSED_DIR), exist_ok=True)
sta_lst = STALTA()


@app.post("/get-quake-startpoint")
async def get_quake_startpoint(file: UploadFile = File(...)):
    
    SAVE_FILE_PATH = "tmp/"
    content = await file.read()

    with open(SAVE_FILE_PATH+file.filename, "wb") as f:
        f.write(content)

    st = read(SAVE_FILE_PATH+file.filename)
    time = sta_lst.get_start(st)
    return JSONResponse({"startPoint": time})

