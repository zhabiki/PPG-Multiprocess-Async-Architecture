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

    disorders = ['checkpoint']
    models = dict()

    for name in disorders:
        model = Inference(
            str(path_models) + '/' + name + '.joblib'
        )
        models[name] = model

    while True:
        if not processed_queue.empty():
            req = processed_queue.get()
            print("main_deamon_sub_inference -> Получены данные из processed_queue.")

            # Чтобы в случае исключения или ошибки, программа не пошла по пятой точке
            try:
                params_all = req[
                    ['bpm', 'sdnn', 'lf/hf']
                ]
                
                predictions = dict()

                for i in range(params_all.shape[0]):
                    params_row = (params_all.iloc[i]).to_numpy()
                    # print(params_row)

                    for name in disorders:
                        try:
                            if predictions.get(name):
                                predictions[name].append(models[name].predict(params_row))
                            else:
                                predictions[name] = [models[name].predict(params_row)]
                        except Exception:
                            if predictions.get(name):
                                predictions[name].append(0.404)
                            else:
                                predictions[name] = [0.404]

                res = dict()
                res = { k: float(np.median(v)) for k, v in predictions.items() }

                print(f'main_deamon_sub_inference -> Итоговые вероятности найдены!')

                if predicted_queue is not None:
                    predicted_queue.put(res)
                    print("main_deamon_sub_inference -> Данные переданы в predicted_queue.")

            except Exception as e:
                print("main_deamon_sub_inference -> Ошибка при обработке данных:", e)

        else:
            time.sleep(1) # Небольшая задержка во избежание перегрузки CPU
