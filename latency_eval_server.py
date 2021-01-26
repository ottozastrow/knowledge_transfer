#!/usr/bin/env python3

import json
import logging
import requests
import socket
import sys
import time
import traceback

from ntrain import train, get_args
import visualization_tools


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def call(args):
    try:
        args, _ = get_args()
        args = vars(args)
        args['disable_wandb'] = True
        args = Struct(**args)
        args.epochs = 0
        args.batch_size = 1
        args.measure_latency = False
        args.connect_rapsi = False
        logging.info("starting training")
        model, test_dataset, batch_size, _ = train(args, None)
        logging.info("starting evaluation")
        latency = visualization_tools.measure_latency(
            model, test_dataset,
            disable_tflite_inference=False, batch_size=batch_size)
        time.sleep(3)
        logging.info("latency is: " + str(latency))
        return latency
    except Exception as err:
        logging.info(err)
        traceback.print_exc()


if __name__ == "__main__":
    logging.basicConfig(filename="logs.txt")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)
    url = "https://knowledgetransfer-30e8a.firebaseio.com"
    response = requests.put(url + "/todo.json", data='{}')
    response = requests.put(url + "/done.json", data='{}')
    logging.info("cleaned firebase db")
    while True:
        try:
            get_response = requests.get(url + "/todo.json")
            assert(get_response.status_code == 200)
            if get_response.text == "null":
                logging.info("waiting for job")
                time.sleep(3.0)
                continue
            logging.info("received job from firebase")
            data = json.loads(get_response.text)
            args = data['args']
            print(args)
            starttime = data['starttime']
            logging.info("starttime is " + str(starttime))
            latency = call(args)
            payload = {'latency': str(latency), 'starttime': starttime,
                       'latency_device': socket.gethostname()}
            payload = json.dumps(payload)
            put_response = requests.put(url + "/todo.json", data='{}')
            put_response = requests.put(url + "/done/" + starttime + ".json",
                                        data=payload)
            logging.info("deleting job and sending latency result to firebase")
            time.sleep(3.0)
            # logging.info("raspi put response", put_response)
        except Exception as err:
            logging.info(err)
            traceback.print_exc()
