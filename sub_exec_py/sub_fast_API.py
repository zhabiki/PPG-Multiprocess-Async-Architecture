from setproctitle import setproctitle
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from multiprocessing import Queue
import uvicorn #  ASGI-сервер, который запускает FastAPI (Asynchronous Server Gateway Interface)

# FastAPI instance
app = FastAPI()

# Очередь будет передана извне
input_queue = None

class LEDData(BaseModel):
    data: List[int]

@app.post("/predict") # эндпойнт - можно поменять
async def predict(data: LEDData):
    print("main_deamon_sub_fast_API -> Принято!")
    if input_queue is not None:
        input_queue.put(data.data)  # передаём данные в очередь
        print("main_deamon_sub_fast_API -> Данные переданы в очередь!")
    return {"received_count": len(data.data)}

def fast_API_run(queue: Queue, host_addres: str = "0.0.0.0", num_port=8000):
    global input_queue
    input_queue = queue

    setproctitle("main_deamon_sub_fast_API")
    print("main_deamon_sub_fast_API -> процесс запущен")

    # Запуск FastAPI-сервера
    uvicorn.run(app, host=host_addres, port=num_port)
