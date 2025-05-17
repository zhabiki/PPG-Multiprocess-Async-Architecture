from setproctitle import setproctitle # type: ignore
import time 


def inference_run(processed_queue, prediction_queue):
    setproctitle("main_deamon_sub_inference")
    print("main_deamon_sub_inference -> процесс запущен")
    time.sleep(5)
    print("Procces_Inference_complete!")