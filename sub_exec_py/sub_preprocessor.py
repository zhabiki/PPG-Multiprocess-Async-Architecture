from setproctitle import setproctitle # type: ignore
import time
import numpy as np

from lib.dataproc.preprocess import PreprocessPPG


def preprocessor_run(input_queue, processed_queue):
    setproctitle("main_deamon_sub_preprocessor")
    print("main_deamon_sub_preprocessor -> ПРОЦЕСС ЗАПУЩЕН")

    preprocess = PreprocessPPG(vis=[
        # 'peaks',
        # 'hrv',
        # 'outliers',
        # 'seg',
    ])

    while True:
        if not input_queue.empty():
            req = input_queue.get()
            print("main_deamon_sub_preprocessor -> Получены данные из input_queue.")
            # print(type(req.data)) # list

            # Чтобы в случае исключения или ошибки, программа не пошла по пятой точке
            try:
                data_np = np.asarray(req.data)
                data_fs = int(req.fs)
                data_wsize = 100
                data_wstride = 10
                data_method = 'noisy'
                data_mode = 'cycles'

                res = preprocess.process_data(
                    data_np[data_fs : len(data_np) - data_fs],
                    data_fs,
                    data_wsize, 
                    data_wstride,
                    data_method,
                    data_mode
                )
                print(f'main_deamon_sub_preprocessor -> Обработка завершена ({res.shape})!')

                if processed_queue is not None:
                    processed_queue.put(res)
                    print("main_deamon_sub_preprocessor -> Данные переданы в processed_queue.")

            except Exception as e:
                print("main_deamon_sub_preprocessor -> Ошибка при обработке данных:", e)

        else:
            time.sleep(1) # Небольшая задержка во избежание перегрузки CPU
