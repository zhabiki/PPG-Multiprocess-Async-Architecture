from setproctitle import setproctitle # type: ignore
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from multiprocessing import Queue
import uvicorn # ASGI-сервер, который запускает FastAPI (Asynchronous Server Gateway Interface)


app = FastAPI() # FastAPI instance
input_queue = None # Очередь будет передана извне

class PredictRequest(BaseModel):
    fs: int
    data: List[int]


@app.post("/predict") # Эндпойнт -- можно поменять
async def predict(data: PredictRequest):
    print("main_deamon_sub_fast_API -> Клиент отправил входные данные!")

    if input_queue is not None:
        input_queue.put(data)
        print("main_deamon_sub_fast_API -> Данные переданы в input_queue.")

    return {"received_count": len(data.data), "received_fs": data.fs} # <-- Д/ОТЛАДКИ


def fast_API_run(queue: Queue, host_addres: str = "0.0.0.0", num_port=8000):
    global input_queue
    input_queue = queue

    setproctitle("main_deamon_sub_fast_API")
    print("main_deamon_sub_fast_API -> ПРОЦЕСС ЗАПУЩЕН")

    # Запуск FastAPI-сервера
    uvicorn.run(app, host=host_addres, port=num_port)
