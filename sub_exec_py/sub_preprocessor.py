from setproctitle import setproctitle # type: ignore
import time
from dataproc import preprocess



def preprocessor_run(input_queue, processed_queue):
    setproctitle("main_deamon_sub_preprocessor")
    print("main_deamon_sub_preprocessor -> процесс запущен")

    # visual = ['find_heartcycle_dists']

    # preprocess_handle = preprocess.PreprocessPPG(vis=visual)

    while True:
        if not input_queue.empty():
            raw_data = input_queue.get()  # Считываем данные из очереди
            print("main_deamon_sub_preprocessor -> получены данные из входящей очереди и направлены на обработку...")


            try: # чтобы в случае исключения или ошибки, программа не пошла по пятой точке
                print(type(raw_data))
                print("main_deamon_sub_preprocessor -> Обработка завершена!")

                # if processed_queue is not None:
                #     processed_queue.put(processed_data)
                #     print("Результат отправлен в выходную очередь.")
            except Exception as e:
                print("Ошибка при обработке данных:", e)

        else:
            time.sleep(0.1)  # Небольшая задержка во избежание перегрузки CPU
    # time.sleep(12)
    # print("Procces_Preprocessor_complete!")