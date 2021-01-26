import ast
import time
import json
import requests


def call_raspberrypi(args, run):
    starttime = str(time.time()).replace(".", '_')
    data = {'args': args, 'starttime': starttime}
    data = json.dumps(data)
    url = "https://knowledgetransfer-30e8a.firebaseio.com"
    print("checking firebase")
    while True:
        time.sleep(1)
        response = requests.get(url + "/todo.json")
        assert(response.status_code == 200), response
        print("waiting for todo to become empty")
        if(response.text == "null"):
            print("todo is empty")
            break

    print("sending job to firebase")
    response = requests.put(url + "/todo.json", data=data)
    response = requests.put(url + "/done/" + starttime + ".json", data='{}')
    assert(response.status_code == 200), response
    counter = 0

    while True:
        if counter == 150:
            raise Exception("timeout for latency result reached. job failed")

        response = requests.get(url + "/done/" + starttime + ".json")
        assert(response.status_code == 200), response
        time.sleep(2)
        if(response.text == "null"):
            counter += 1
            print(counter, "/150 - waiting for latency result")
            continue
        respond_data = ast.literal_eval(response.text)

        print("respond data is ", (respond_data))
        if respond_data['latency'] == "None":
            raise Exception("eval server crashed")
        latency = float(respond_data["latency"])
        assert(respond_data["starttime"] == starttime)
        run.summary['latency_device'] = respond_data["latency_device"]
        response = requests.put(url + "/done/" + starttime + ".json",
                                data='{}')

        run.summary['latency'] = latency
        run.summary.update()
        assert(latency <= 1.0)
        break
