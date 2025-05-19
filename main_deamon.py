import time

from sub_exec_py import sub_fastAPI, sub_preprocessor, sub_inference
from multiprocessing import Process, Queue


def start_process(target, name, args):
    p = Process(target=target, name=name, args=args)
    p.start()
    return p


if __name__ == "__main__":
    # Собственно, сами очереди для обмена
    input_queue = Queue()
    processed_queue = Queue()
    predicted_queue = Queue()

    host_address = "0.0.0.0" # Задаём IP адресс сервера sub_fastAPI
    num_port = 8000 # Задаём порт сервера sub_fastAPI

    processes = {
        "fastAPI": start_process(sub_fastAPI.fastAPI_run, "FastAPI", (input_queue, predicted_queue, host_address, num_port)),
        "preprocessor": start_process(sub_preprocessor.preprocessor_run, "Preprocessor", (input_queue, processed_queue)),
        "inference": start_process(sub_inference.inference_run, "Inference", (processed_queue, predicted_queue))
    }

    try:
        while True:
            for name, process in list(processes.items()):
                if not process.is_alive():
                    print(f"main_deamon -> Подпроцесс '{name}' упал!!! Перезапускаю...")
                    # Перезапускаем упавшие подпроцессы
                    if name == "fastAPI":
                        processes[name] = start_process(sub_fastAPI.fastAPI_run, "FastAPI", (input_queue, predicted_queue, host_address, num_port)),
                    elif name == "preprocessor":
                        processes[name] = start_process(sub_preprocessor.preprocessor_run, "Preprocessor", (input_queue, processed_queue))
                    elif name == "inference":
                        processes[name] = start_process(sub_inference.inference_run, "Inference", (processed_queue, predicted_queue))
            time.sleep(1) # Чтобы не грузить ЦП

    except KeyboardInterrupt:
        print("\nmain_deamon -> Завершаю все подпроцессы...")
        for p in processes.values():
            p.terminate()
            p.join()
