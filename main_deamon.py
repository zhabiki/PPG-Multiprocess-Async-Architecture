import time

from sub_exec_py import sub_fast_API, sub_preprocessor, sub_inference
from multiprocessing import Process, Queue

def start_process(target, name, args):
    p = Process(target=target, name=name, args=args)
    p.start()
    return p

if __name__ == "__main__":
    # собственно сами очереди для обмена
    input_queue = Queue()
    processed_queue = Queue()
    prediction_queue = Queue()

    host_addres = "0.0.0.0" # задаем IP адресс сервера sub_fast_API
    num_port = 8000 # задаем порт сервера sub_fast_API

    processes = {
        "fast_API": start_process(sub_fast_API.fast_API_run, "FastAPI", (input_queue, host_addres, num_port)),
        "preprocessor": start_process(sub_preprocessor.preprocessor_run, "Preprocessor", (input_queue, processed_queue)),
        "inference": start_process(sub_inference.inference_run, "Inference", (processed_queue, prediction_queue))
    }

    try:
        while True:
            for name, process in list(processes.items()):
                if not process.is_alive():
                    print(f"[!] Process is named '{name}' was crached! Restart process...")
                    # перезапускаем упавшие процессы
                    if name == "fast_API":
                        processes[name] = start_process(sub_fast_API.fast_API_run, "FastAPI", (input_queue,))
                    elif name == "preprocessor":
                        processes[name] = start_process(sub_preprocessor.preprocessor_run, "Preprocessor", (input_queue, processed_queue))
                    elif name == "inference":
                        processes[name] = start_process(sub_inference.inference_run, "Inference", (processed_queue, prediction_queue))
            time.sleep(1)  # чтобы не грузить ЦП
    except KeyboardInterrupt:
        print("\n[*] Завершаем все процессы...")
        for p in processes.values():
            p.terminate()
            p.join()








# # import sys
# # import os

# from sub_exec_py import sub_fast_API, sub_preprocessor, sub_inference
# from multiprocessing import Process, Queue


# if __name__ == "__main__":
#     # Очереди для обмена данными между процессами
#     input_queue = Queue()
#     processed_queue = Queue()
#     prediction_queue = Queue()

#     # Запуск процессов
#     p1 = Process(target=sub_fast_API.fast_API_run, args=(input_queue,))
#     p2 = Process(target=sub_preprocessor.preprocessor_run, args=(input_queue, processed_queue))
#     p3 = Process(target=sub_inference.inference_run, args=(processed_queue, prediction_queue))

#     p1.start()
#     p2.start()
#     p3.start()

#     p1.join()
#     p2.join()
#     p3.join()
