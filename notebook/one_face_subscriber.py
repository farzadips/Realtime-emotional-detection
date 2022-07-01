import datetime
from DoSomething import DoSomething
import time
import json

import tensorflow as tf
import numpy as np



class Subscriber(DoSomething):
    def notify(self, topic, msg):
        input_json = json.loads(msg)
        
        datetime = input_json['bt']

        state = input_json['e']

        
 

        print('Datetime = {}'.format(datetime))
        print('current state = {}'.format(state))
        print('\n')



if __name__ == "__main__":
    test = Subscriber("subscriber 1")
    test.run()
    test.myMqttClient.mySubscribe("/206803/emotion")

    while True:
        time.sleep(1)
