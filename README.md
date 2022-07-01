# Realtime-emotional-detection
The main aim of this program and device is to be an assistant for finding the emotions of people in different situations to figure out the best ways to handle problems or getting wise decisions like in shops or in public areas since the emotions of people represents their way of thinking about a case. This program is able to identify different human emotions in situations where there is more than 1 person in the camera field of view with different backgrounds. 


We used a publisher on the raspberry side to transfer the emotions from model to the client, there would be a subscriber for receiving the emotions for multiple users. We have 2 subscriber 1 for detecting 1 face and another for detectiong multiple faces (Fig 10) The reason we chose MQTT was regarding sending multiple messages at high speed at the same time.
