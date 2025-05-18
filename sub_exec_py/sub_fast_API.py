from setproctitle import setproctitle # type: ignore
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from multiprocessing import Queue
import uvicorn # ASGI-сервер, который запускает FastAPI (Asynchronous Server Gateway Interface)
import threading
import time


app = FastAPI() # FastAPI instance
input_queue = None # Очередь будет передана извне
result_queue = None

inference_results = []

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


@app.get("/predict")
async def get_prediction():
    print("main_deamon_sub_fast_API -> Клиент запрашивает результат предсказания!")

    if inference_results:
        result = inference_results.pop(0)
        print(f"main_deamon_sub_fast_API -> Отправлен результат: {result}")
        return {"result": result}
    else:
        print("main_deamon_sub_fast_API -> Результатов пока нет.")
        return {"error": "Результатов пока нет"}


def result_listener():
    print("main_deamon_sub_fast_API -> Фоновый поток для чтения из очереди инференса запущен.")
    while True:
        if result_queue is not None:
            try:
                result = result_queue.get(timeout=1)  
                inference_results.append(result)
                print(f"main_deamon_sub_fast_API -> Получен результат: {result}")
            except:
                pass  # timeout или очередь пуста — продолжаем типа ждать ЫЫЫЫЫ
        else:
            time.sleep(0.5)


def fast_API_run(in_queue: Queue, out_queue: Queue, host_addres: str = "0.0.0.0", num_port=8000):
    global input_queue, result_queue
    input_queue, result_queue = in_queue, out_queue

    setproctitle("main_deamon_sub_fast_API")
    print("main_deamon_sub_fast_API -> ПРОЦЕСС ЗАПУЩЕН")

    thread = threading.Thread(target=result_listener, daemon=True)
    thread.start()

    # Запуск FastAPI-сервера
    uvicorn.run(app, host=host_addres, port=num_port)
