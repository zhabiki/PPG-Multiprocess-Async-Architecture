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
predicted_queue = None # Очередь будет передана извне

inference_results = []

class PredictRequest(BaseModel):
    fs: int
    data: List[int]


@app.post("/predict") # Эндпойнт -- можно поменять
async def predict(data: PredictRequest):
    print("main_deamon_sub_fastAPI -> Клиент отправил входные данные!")

    if (len(data.data) > 500000) or (len(data.data) / data.fs < 30.0):
        return {"error": "Данные ФПГ слишком длинные или слишком короткие."}

    if input_queue is not None:
        input_queue.put(data)
        print("main_deamon_sub_fastAPI -> Данные переданы в input_queue.")
    # return {"received_count": len(data.data), "received_fs": data.fs}

    timeout_cnt = 0
    while not inference_results and timeout_cnt < 5:
        time.sleep(3)
        timeout_cnt += 1

    if inference_results:
        result = inference_results.pop(0)
        print(f"main_deamon_sub_fastAPI -> Отправлен результат: {result}")
        return {"result": result}
    else:
         return {"error": "Превышено время ожидания обработки данных."}


@app.get("/predict")
async def get_prediction():
    print("main_deamon_sub_fastAPI -> Клиент запрашивает результат предсказания!")

    if inference_results:
        result = inference_results.pop(0)
        print(f"main_deamon_sub_fastAPI -> Отправлен результат: {result}")
        return {"result": result}
    else:
        print("main_deamon_sub_fastAPI -> Результатов пока нет.")
        return {"error": "Результатов пока нет"}


def result_listener():
    print("main_deamon_sub_fastAPI -> Фоновый поток для чтения из очереди инференса запущен.")
    while True:
        if predicted_queue is not None:
            try:
                result = predicted_queue.get(timeout=1)  
                inference_results.append(result)
                print(f"main_deamon_sub_fastAPI -> Получен результат: {result}")
            except:
                pass # Timeout или очередь пуста -- продолжаем типа ждать ЫЫЫЫЫЫ
        else:
            time.sleep(0.5)


def fastAPI_run(in_queue: Queue, out_queue: Queue, host_address: str = "0.0.0.0", num_port=8000):
    global input_queue, predicted_queue
    input_queue, predicted_queue = in_queue, out_queue

    setproctitle("main_deamon_sub_fastAPI")
    print("main_deamon_sub_fastAPI -> ПРОЦЕСС ЗАПУЩЕН")

    thread = threading.Thread(target=result_listener, daemon=True)
    thread.start()

    # Запуск FastAPI-сервера
    uvicorn.run(app, host=host_address, port=num_port)
