import datetime
from DoSomething import DoSomething
import time
import json

import tensorflow as tf
import numpy as np



class Subscriber(DoSomething):
    def notify(self, topic, msg):
        input_json = json.loads(msg)
        print(input_json)
        datetime = input_json['bt']

        state_1 = input_json['fe']
        state_2 = input_json['se']

        
 

        print('Datetime = {}'.format(datetime))
        print('first emotion = {}'.format(state_1))
        print('second emotion = {}'.format(state_2))
        print('\n')



if __name__ == "__main__":
    test = Subscriber("subscriber 2")
    test.run()
    test.myMqttClient.mySubscribe("/206803/multi_emotion")

    while True:
        time.sleep(1)
