"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import pdb
import argparse
import collections
import os
import sys
import numpy as np
import csv
import time
import socket
import json
import cv2
from sys import platform
import logging as log
import paho.mqtt.client as mqtt

# from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

#INPUT_STREAM = "pets.mp4"
# Get correct CPU extension
if platform == "linux" or platform == "linux2":
    CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
    CODEC = 0x00000021
elif platform == "darwin":
    CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib"
    CODEC = cv2.VideoWriter_fourcc('M','J','P','G')
else:
    exit(1)

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = argparse.ArgumentParser("Basic Edge App with Inference Engine")
    #parser = ArgumentParser("Basic Edge App with Inference Engine")
#     i_desc = "The location of the input image"
#     m_desc = "The location of the model XML file"
#     parser._action_groups.pop()
#     required = parser.add_argument_group('required arguments')
#     optional = parser.add_argument_group('optional arguments')
    parser.add_argument("-m","--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input",required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.6,
                        help="Probability threshold for detections filtering"
                        "(0.6 by default)")
    parser.add_argument("-pc", "--perf_counts", type = str, default= False, help = "Print performance counters")
    args = parser.parse_args()
    return args

def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client= mqtt.Client()
    client.connect(MQTT_HOST,MQTT_PORT,MQTT_KEEPALIVE_INTERVAL)
    return client

### performance countes
def performance_counts(perf_count):
    """
    print information about layers of the model.
    :param perf_count: Dictionary consists of status of the layers.
    :return: None
    """
    log.info("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type',
                                                      'exec_type', 'status',
                                                      'real_time, us'))
    for layer, stats in perf_count.items():
        log.info("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer,
                                                          stats['layer_type'],
                                                          stats['exec_type'],
                                                          stats['status'],
                                                          stats['real_time']))
    
    
###access the data in the scene
def assess_scene(a,personexist,prob_threshold):        
        movingrange = a
        if movingrange >prob_threshold and personexist == False:                                 
            personexist = True            
        elif movingrange<prob_threshold and personexist == True:            
            personexist = False            
        return personexist

def draw_boxes(personexist,frame, result, args,width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    current_count = 0
    # Output shape is 1x1x100x7
    if personexist:
        box = result[0][0][0]
        xmin = int(box[3] * width)
        ymin = int(box[4] * height)
        xmax = int(box[5] * width)
        ymax = int(box[6] * height)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
        current_count +=1         
    return frame, current_count

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    request_id=0
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, request_id,args.device, CPU_EXTENSION) 
    net_input_shape = infer_network.get_input_shape(request_id)
    
    ### TODO: Handle the input stream ###
    image_flag = False
   
    if args.input == "CAM":
        args.input = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        image_flag = True
        input_stream = args.input
    elif args.input.endswith('.mp4') or args.input.endswith('.avi'):
        input_stream = args.input
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"
    
    cap = cv2.VideoCapture(args.input)
    if input_stream: 
        cap.open(args.input)
    if not cap.isOpened():
        log.error("ERROR! Unable to open video source")    
    
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    ### initialize timer counter
    t = 0
    start = 0
    last_count = 0
    ### initialize incident_flag,  total_count, duration
    personexist = False
    total_count,duration=0,0
       
    ### TODO: Loop until stream is over ###
    a=collections.deque([0 for _ in range(20)])
    b=np.array([0])
    c=np.array([0])
#    infer_start = time.time()
    while cap.isOpened():
       
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
            
        key_pressed = cv2.waitKey(60)
        t +=1
        # Pre-process the frame
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
#        infer_start = time.time()
        infer_network.exec_net(request_id,p_frame)
        ### TODO: Wait for the result ###
        if infer_network.wait(request_id) == 0:
#            det_time = time.time() - infer_start
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output(request_id)
            confidence = result[0][0][0][2]
#             a.append(confidence)
#             a.popleft()
            b=np.append(b,[confidence],axis = 0)
    ### take the last 20 frame confidence, drop 15 smallest, only use 5 highest confidence value to determine the status
            newa = b[-20:]
            newb = sorted(newa,reverse = True)
            newc = newb[:5]
            new = sum(newc)/5
            
#            c=np.append(c,[new],axis = 0)
#             print(b,c)
#             pdb.set_trace()
#             if args.perf_counts:
#                 perf_count = infer_network.performance_counter()
#                 performance_counts(perf_count)
            
            ### TODO: Extract any desired stats from the results ###
            personexist = assess_scene(new,personexist,prob_threshold) 
            frame,current_count = draw_boxes(personexist,frame,result,args, width, height)
#             ### TODO: Calculate and send relevant information on ###
#             ### current_count, total_count and duration to the MQTT server ###
#             ### Topic "person": keys of "count" and "total" ###
#             ### Topic "person/duration": key of "duration" ###
            # When new person enters the video
            if current_count > last_count:
                 start_time = time.time()
                 total_count = total_count + current_count - last_count
                 client.publish("person", json.dumps({"total": total_count}))

            # Person duration in the video is calculated
            if current_count < last_count:
                 duration = int(time.time() - start_time)
                 # Publish messages to the MQTT server
                 client.publish("person/duration",
                                json.dumps({"duration": duration}))

            client.publish("person", json.dumps({"count": current_count}))
            last_count = current_count
              
        ### TODO: Send the frame to the FFMPEG server ###      
        image = np.uint8(frame)        
        sys.stdout.buffer.write(image)
        sys.stdout.flush()
                
        # Break if escape key pressed
        if key_pressed == 27:
            break
            
        ### TODO: Write an output image if `single_image_mode` ###
        if image_flag:
            cv2.imwrite('output_image.jpg',frame) 
    ### output all confidence values in foo.csv
#    np.savetxt("b.csv",b,delimiter=",")
#    np.savetxt("pre.csv", c, delimiter=",")
    ### output the inference time
#    det_time = time.time() - infer_start
#    print("infer time of this model:{:.2f}s".format(det_time))
    cap.release()
    cv2.destroyAllWindows()
    ### TODO: Disconnect from MQTT
    client.disconnect()
    infer_network.clean()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser()
    # Connect to the MQTT server
    
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
