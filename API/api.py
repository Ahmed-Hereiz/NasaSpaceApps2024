from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import json
import os
import time
import yaml
import httpx

app = FastAPI()

UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.join(UPLOAD_DIR, PROCESSED_DIR), exist_ok=True)


@app.get("/")
def home():
    return {"message": "Hello, From MoonQuake API!"}


@app.post("/process-chunk")
async def process_chunk(chunk: UploadFile = File(...)):
    try:

        processed_chunk = await chunk.read()

        return Response(
            content=processed_chunk,
            media_type="application/octet-stream"
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/get-quake-startpoint")
def get_quake_startpoint(file: UploadFile = File(...)):
    """
    processed_content = b""
    
    async with httpx.AsyncClient() as client:
        while True:
            # read the file in chunks of 8192 bytes
            chunk = await file.read(8192)
            
            if not chunk:
                break
            
            # send the chunk to the processing endpoint
            response = await client.post(
                "http://127.0.0.1:8080/process-chunk",
                files={"chunk": ("chunk", chunk)},
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Failed to process chunk"
                )

            # append the processed chunk to our result
            processed_content += response.content

    output_path = os.path.join(os.path.join(
        UPLOAD_DIR, PROCESSED_DIR), f"processed_{file.filename}")
    
    with open(output_path, "wb") as f:
        f.write(processed_content)
        
     return FileResponse(
        path=output_path,
        filename=f"processed_{file.filename}",
        media_type=file.content_type
    )
        """
    return JSONResponse({"startPoint": 2})


class ChartRequest(BaseModel):
    startPoint: str


@app.post("/get-chart-from-quake-startpoint")
def get_chart_from_quake_startpoint(request: ChartRequest):
    return JSONResponse({
        "image_url": f"image_url/{request.startPoint}"
    })

@app.post("/get-upload-download")
async def get_response_download(file: UploadFile = File(...)):
    try:
        content = await file.read()
        # add time stamp
        timeStr = time.strftime("%Y%m%d-%H%M%S")
        new_filename = "{}_{}.txt".format(
            os.path.splitext(file.filename)[0], timeStr)
        
        SAVE_FILE_PATH = os.path.join(UPLOAD_DIR, new_filename)
        
        # Determine the file type and process accordingly
        if file.content_type == 'application/json':
            json_data = json.loads(content)
            with open(SAVE_FILE_PATH, "w") as f:
                f.write(json_data)
        else:
            with open(SAVE_FILE_PATH, "wb") as f:
                f.write(content)
                
        return FileResponse(path=SAVE_FILE_PATH, media_type="application/octet-stream", filename=new_filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

