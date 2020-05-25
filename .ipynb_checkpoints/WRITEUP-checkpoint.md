# Project Write-Up

## Explaining Custom Layers

(Ref to info https://docs.openvinotoolkit.org/latest/_docs_HOWTO_Custom_Layers_Guide.html)
The Intel® Distribution of OpenVINO™ toolkit supports neural network model layers in multiple frameworks including TensorFlow*, Caffe*, MXNet*, Kaldi* and ONYX*. The list of known layers is different for each of the supported frameworks. To see the layers supported by your framework, refer to https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html.
Custom layers are layers that are not included in the list of known layers. If your topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom.
The Model Optimizer searches the list of known layers for each layer contained in the input model topology before building the model's internal representation, optimizing the model, and producing the Intermediate Representation files.
The Inference Engine loads the layers from the input model IR files into the specified device plugin, which will search a list of known layer implementations for the device. If your topology contains layers that are not in the list of known layers for the device, the Inference Engine considers the layer to be unsupported and reports an error. If a device doesn't support a particular layer, an alternative to creating a new custom layer is to target an additional device using the HETERO plugin. The Heterogeneous Plugin may be used to run an inference model on multiple devices allowing the unsupported layers on one device to "fallback" to run on another device (e.g., CPU) that does support those layers.
Custom Layer Implementation Workflow
When implementing a custom layer for your pre-trained model in the Intel® Distribution of OpenVINO™ toolkit, you will need to add extensions to both the Model Optimizer and the Inference Engine.

Custom Layer Extensions for the Model Optimizer
The following figure shows the basic processing steps for the Model Optimizer highlighting the two necessary custom layer extensions, the Custom Layer Extractor and the Custom Layer Operation.


The Model Optimizer first extracts information from the input model which includes the topology of the model layers along with parameters, input and output format, etc., for each layer. The model is then optimized from the various known characteristics of the layers, interconnects, and data flow which partly comes from the layer operation providing details including the shape of the output for each layer. Finally, the optimized model is output to the model IR files needed by the Inference Engine to run the model.

The Model Optimizer starts with a library of known extractors and operations for each supported model framework which must be extended to use each unknown custom layer. The custom layer extensions needed by the Model Optimizer are:

Custom Layer Extractor
Responsible for identifying the custom layer operation and extracting the parameters for each instance of the custom layer. The layer parameters are stored per instance and used by the layer operation before finally appearing in the output IR. Typically the input layer parameters are unchanged, which is the case covered by this tutorial.
Custom Layer Operation
Responsible for specifying the attributes that are supported by the custom layer and computing the output shape for each instance of the custom layer from its parameters.
The --mo-op command-line argument shown in the examples below generates a custom layer operation for the Model Optimizer.
Custom Layer Extensions for the Inference Engine
The following figure shows the basic flow for the Inference Engine highlighting two custom layer extensions for the CPU and GPU Plugins, the Custom Layer CPU extension and the Custom Layer GPU Extension.


Each device plugin includes a library of optimized implementations to execute known layer operations which must be extended to execute a custom layer. The custom layer extension is implemented according to the target device:

Custom Layer CPU Extension
A compiled shared library (.so or .dll binary) needed by the CPU Plugin for executing the custom layer on the CPU.
Custom Layer GPU Extension
OpenCL source code (.cl) for the custom layer kernel that will be compiled to execute on the GPU along with a layer description file (.xml) needed by the GPU Plugin for the custom layer kernel.
Model Extension Generator
Using answers to interactive questions or a *.json* configuration file, the Model Extension Generator tool generates template source code files for each of the extensions needed by the Model Optimizer and the Inference Engine. To complete the implementation of each extension, the template functions may need to be edited to fill-in details specific to the custom layer or the actual custom layer functionality itself.

Command-line
The Model Extension Generator is included in the Intel® Distribution of OpenVINO™ toolkit installation and is run using the command (here with the "--help" option):

python3 /opt/intel/openvino/deployment_tools/tools/extension_generator/extgen.py new --help
where the output will appear similar to:

usage: You can use any combination of the following arguments:
Arguments to configure extension generation in the interactive mode:
optional arguments:
  -h, --help            show this help message and exit
  --mo-caffe-ext        generate a Model Optimizer Caffe* extractor
  --mo-mxnet-ext        generate a Model Optimizer MXNet* extractor
  --mo-tf-ext           generate a Model Optimizer TensorFlow* extractor
  --mo-op               generate a Model Optimizer operation
  --ie-cpu-ext          generate an Inference Engine CPU extension
  --ie-gpu-ext          generate an Inference Engine GPU extension
  --output_dir OUTPUT_DIR
                        set an output directory. If not specified, the current
                        directory is used by default.
The available command-line arguments are used to specify which extension(s) to generate templates for the Model Optimizer or Inference Engine. The generated extension files for each argument will appear starting from the top of the output directory as follows:

COMMAND-LINE ARGUMENT	OUTPUT DIRECTORY LOCATION
--mo-caffe-ext	user_mo_extensions/front/caffe
--mo-mxnet-ext	user_mo_extensions/front/mxnet
--mo-tf-ext	user_mo_extensions/front/tf
--mo-op	user_mo_extensions/ops
--ie-cpu-ext	user_ie_extensions/cpu
--ie-gpu-ext	user_ie_extensions/gpu
Extension Workflow
The workflow for each generated extension follows the same basic steps:

Step 1: Generate: Use the Model Extension Generator to generate the Custom Layer Template Files.

Step 2: Edit: Edit the Custom Layer Template Files as necessary to create the specialized Custom Layer Extension Source Code.

Step 3: Specify: Specify the custom layer extension locations to be used by the Model Optimizer or Inference Engine.

Some of the potential reasons for handling custom layers: writing your own layer object enables you to support the drawing of new data formats and to customize how existing data formats display. A custom layer allows you to display an unsupported data source in a map. You could also change the way an existing layer class draws by using a custom layer.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were by comparing the model accuracy, size and inference time before and after conversion. 

The difference between model accuracy pre- and post-conversion was judged by corrected detection frames/total frames. Corrected detection means output 1 when the frame has person, and 0 when the frame has no person. 

The size of the model pre- was the size of model file, and post-conversion is the size of xml file.

The inference time of the model pre- and post-conversion was compared directly using the inference time. 
model	size	time
ssd_mobilenet_v2_coco_2018_03_29	200MB	184.78s
pedestrian-detection-adas-binary-0001	2.33MB	137.75s
After performing frame dropping on models, both model could provide correct result. For ssd_mobilenet_v2_coco_2018_03_29, the threshold is 0.15. Raising the threshold will result in false positive (more than actual people would be counted). And for intel model, the threshold is 0.8. Lowering the threshold will result in false positive. And when observing the bounding box on the frame, the bounding box in ssd model is more stable from frame to frame, while the bounding box in intel model jumps from frame to frame.
If the use case is to control the population in a confined space (shopping mall, theater), false positive is safer, I would select ssd_mobilenet_v2_coco_2018_03_29 with a little higher threshold. The size of the model should not be a problem for such case, since large device could be used in those cases. 
If the use case is home camera monitoring or outdoor monitoring, where only small devices are available, I would select intel model due to the size of the model and the accuracy. 


## Assess Model Use Cases

Some of the potential use cases of the people counter app are 
1. controlling the population of people in confined space, such as malls, shopping centers, during pandemic; one app at the entrance and one app at the exit could provide how many people are inside the space. If the population inside reach the limit, no new person is allowed to enter. 
2. determine the business strategy(resources distribution at different time period and business units) based on the peak time of population and duration for a special business, such as hotel registration, resturants; for example in a full service hotel, one app at the registration area could tell guests like to registrate around 4pm and one app at the dinning area could tell guest like to come to dinner at 6pm. With those information, the hotel could have more people work in registration table from 4pm to 6pm and those people could start to work in the dinning module after 6pm.

Each of these use cases would be useful because this app provide in-situ statistic number of people through a physical space. It provides information about how many people in/out the space. Those information could be used to direct control the population, aslo could be used to determine how much resources should be invested to serve those people in a certain time and certain business unit.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows:
	• Poor lighting will result in dramatical failure regarding model's accuracy or even unusable. However, pre-process image of each frame (such as normalizing the image) with proper brightness/contrast adjustment could resolve this before passing it to the model.
	• Model accuracy natural decreasing during conversion or other stages may make the model unusable if the doesn't perform the required task such as detecting risk of crimes as mentioned above in use-cases. A effective solution to this would be to put the model into validation mode for an hour or so. During this time the various devices could perform federated learning to give better performance. This might drastically provide a better performance.
camera focal length/image size will affect the model because the model may fail to make sense of the input and the distored input may not be detected properly by the model. An approach to solve this would be to use similar images(similar focal length and image size) while training models  for use cases and specifying the threshold skews, this could be a potential solution. If use cases changes over time, the model needs to be training again. 

## Model Research


In investigating potential people counter models, I tried each of the following three models:

Model 1: MobileNetSSD_deploy10695.caffemodel

I get this model by
  git clone https://github.com/zlingkang/mobilenet_ssd_pedestrian_detection

I converted the model to an Intermediate Representation with the following arguments:
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model mobilenet_ssd_pedestrian_detection/MobileNetSSD_deploy10695.caffemodel --input_proto mobilenet_ssd_pedestrian_detection/MobileNetSSD_deploy.prototxt -o /home/workspace/intel/
  
   The model was insufficient for the app because it did not recognize peoson in the frame in most frames, confidence in most frames are 0. I tried to improve the model detection rate by smoothing the confidence by moving range. I outputed the confidence of each frame in  and saved in mobilenet_ssd_deploy10695.xlsx. 
   Moving range of 20 doesn't work since too many 0s of confidence. 
    
- Model 2: mobilenet_iter_73000.caffemodel
Git clone
https://github.com/chuanqi305/MobileNet-SSD
I converted the model to an Intermediate Representation with the following arguments
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model MobileNet-SSD/mobilenet_iter_73000.caffemodel --input_proto MobileNet-SSD/deploy.prototxt -o /home/workspace/intel
    
The model was insufficient for the app because it detected too many false positive that could not be filter out.I tried to improve this model by filtering out some false positive. I outputed the confidence of each frame in  and saved in mobilenet_iter_73000.xlsx. From the result there are too many 1s at the begining and the end of the video. I could filter out single point false positive by moving range method of confidence, but When there are continuous false positive for more than 20 points, it doesn't work anymore.
  
- Model 3: ssd_mobilenet_v2_coco_2018_03_29
I get this model by 
    wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz 
    tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz

  - I converted the model to an Intermediate Representation with the following arguments
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  - The model was insufficient at the begining because there are some false negative in the middle of the video. I tried to improve this model by filtering out some false negative. I outputed the confidence of each frame in  and saved in frozen_inference_graph.xlsx. Actually this one works after taking 20 frames moving average confidence and use 0.04 as the threshold. It could detect exactly 6 peolpe in this video and output the correct duration for each person. But theshold 0.04 is kind of too low for a general cases. This model won't work for other videos. 
  To improve this model, I took the last 20 frame confidence, drop 15 smallest, only use 5 highest confidence value to determine the status. After performing frame dropping on models, model could provide correct result with the threshold 0.15. Raising the threshold will result in false positive (more than actual people would be counted). The new confidence was saved in confidence after framedroping for frozen model.xlsx.
  
  
