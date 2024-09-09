import os
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from skimage.exposure import match_histograms
import numpy as np
import cv2
import matplotlib.pyplot as plt

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates directory
templates = Jinja2Templates(directory="templates")

# Ensure necessary directories exist
if not os.path.exists("static/uploads"):
    os.makedirs("static/uploads")

if not os.path.exists("static/histograms"):
    os.makedirs("static/histograms")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/bitwise_and/", response_class=HTMLResponse)
async def bitwise_and_operation(request: Request, file1: UploadFile = File(...), file2: UploadFile = File(...)):
    # Load the two images
    image_data1 = await file1.read()
    np_array1 = np.frombuffer(image_data1, np.uint8)
    frame_1 = cv2.imdecode(np_array1, cv2.IMREAD_COLOR)

    image_data2 = await file2.read()
    np_array2 = np.frombuffer(image_data2, np.uint8)
    frame_2 = cv2.imdecode(np_array2, cv2.IMREAD_COLOR)

    # Perform bitwise AND operation
    result_image = cv2.bitwise_and(frame_1, frame_2)

    # Save the resulting image
    output_filename = f"frame_{uuid4().hex[:8]}.png"
    output_path = os.path.join("static/uploads", output_filename)
    cv2.imwrite(output_path, result_image)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "modified_image_path": f"/static/uploads/{output_filename}"
    })

@app.post("/add_logo/", response_class=HTMLResponse)
async def add_logo(request: Request, picture_file: UploadFile = File(...), logo_file: UploadFile = File(...)):
    # Load the picture frame and the logo
    picture_data = await picture_file.read()
    np_picture = np.frombuffer(picture_data, np.uint8)
    picture_frame = cv2.imdecode(np_picture, cv2.IMREAD_COLOR)

    logo_data = await logo_file.read()
    np_logo = np.frombuffer(logo_data, np.uint8)
    logo_polban = cv2.imdecode(np_logo, cv2.IMREAD_UNCHANGED)

    # Resize the logo
    scale_percent = 70
    width = int(logo_polban.shape[1] * scale_percent / 100)
    height = int(logo_polban.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_logo = cv2.resize(logo_polban, dim, interpolation=cv2.INTER_AREA)

    # Split channels and mask
    b, g, r, alpha = cv2.split(resized_logo)
    mask = alpha

    # Determine position to place the resized logo
    x_offset = picture_frame.shape[1] // 2 - resized_logo.shape[1] // 2
    y_offset = picture_frame.shape[0] // 2 - resized_logo.shape[0] // 2

    # Blend the logo with the frame
    roi = picture_frame[y_offset:y_offset+resized_logo.shape[0], x_offset:x_offset+resized_logo.shape[1]]
    foreground = cv2.merge((b, g, r))
    background = roi
    blended = cv2.add(cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask)), 
                      cv2.bitwise_and(foreground, foreground, mask=mask))

    # Place the blended image back into the original frame
    result_frame = picture_frame.copy()
    result_frame[y_offset:y_offset+resized_logo.shape[0], x_offset:x_offset+resized_logo.shape[1]] = blended

    # Save the resulting image
    output_filename = f"logo_frame_{uuid4().hex[:8]}.png"
    output_path = os.path.join("static/uploads", output_filename)
    cv2.imwrite(output_path, result_frame)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "modified_image_path": f"/static/uploads/{output_filename}"
    })

@app.post("/add_text/", response_class=HTMLResponse)
async def add_text(request: Request, file: UploadFile = File(...), text: str = Form(...)):
    # Load the image
    image_data = await file.read()
    np_image = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # Add text to the image
    text_coordinates = (175, 531)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 255)
    line_type = 2

    cv2.putText(img, text, text_coordinates, font, font_scale, font_color, line_type)

    # Save the resulting image
    output_filename = f"final_{uuid4().hex[:8]}.png"
    output_path = os.path.join("static/uploads", output_filename)
    cv2.imwrite(output_path, img)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "modified_image_path": f"/static/uploads/{output_filename}"
    })

def save_image(image, prefix):
    filename = f"{prefix}_{uuid4()}.png"
    path = os.path.join("static/uploads", filename)
    cv2.imwrite(path, image)
    return f"/static/uploads/{filename}"
