python -m venv venv

venv\\Scripts\\activate

pip install fastapi uvicorn jinja2 opencv-python-headless python-multipart sqlalchemy databases sqlite-utils  

uvicorn main:app --reload

PS D:\Semester 5\PCD\Praktek\w1 - init\fastapi-opencv-profile> uvicorn main:app --reload
>>
INFO:     Will watch for changes in these directories: ['D:\\Semester 5\\PCD\\Praktek\\w1 - init\\fastapi-opencv-profile']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [9156] using StatReload
INFO:     Started server process [8324]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     127.0.0.1:62930 - "GET / HTTP/1.1" 200 OK
INFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [8324]
INFO:     Stopping reloader process [9156]

\
folder 
static\uploads
static\histogram
