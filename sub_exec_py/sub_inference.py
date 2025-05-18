from setproctitle import setproctitle # type: ignore
import time 
import numpy as np
from pathlib import Path

from lib.models.inference import Inference


def inference_run(processed_queue, predicted_queue):
    setproctitle("main_deamon_sub_inference")
    print("main_deamon_sub_inference -> ПРОЦЕСС ЗАПУЩЕН")

    path_curr = Path(__file__).parent.resolve()
    path_models = path_curr.parent / 'lib' / 'models' / 'saves'
    # print(path_models)

    disorders = ['anxiety', 'bpad', 'depression', 'none']
    ensemble = Inference(
        [str(path_models / f'{name}_model.json') for name in disorders],
    disorders)

    while True:
        if not processed_queue.empty():
            req = processed_queue.get()
            print("main_deamon_sub_inference -> Получены данные из processed_queue.")

            # Чтобы в случае исключения или ошибки, программа не пошла по пятой точке
            try:
                params_all = req[
                    ['bpm', 'sdnn', 'lf/hf']
                ]

                preds = ensemble.ensemble.forward(params_all.to_numpy(), probas=True)
                # print(preds)

                preds_mean = {
                    d: float(np.median([e[d][1] for e in preds]))
                    for d in disorders
                }
                print(f'main_deamon_sub_inference -> Итоговые вероятности найдены!')

                if predicted_queue is not None:
                    predicted_queue.put(preds_mean)
                    print("main_deamon_sub_inference -> Данные переданы в predicted_queue.")

            except Exception as e:
                print("main_deamon_sub_inference -> Ошибка при обработке данных:", e)

        else:
            time.sleep(1) # Небольшая задержка во избежание перегрузки CPU
