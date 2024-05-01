# Pothole Detection Optimization
# Project Description
Pothole detection systems offer a promising solution to address the negative effects of road degradation. Therefore, we decided to optimize the inference time of a YOLOv8 model for pothole detection on a Raspberry Pi 4 by using transfer learning and optimization techniques. 
# Implementation on the Raspberry Pi 4
As for the embedded device implementation, we implemented our model on a Raspberry Pi 4 model B which has the following specifications: 64 bit quad-core Cortex-A72 processor and 4 GB LPDDR4 RAM. This model offers enhanced computational power and memory capacity, adequate for running resource-intensive applications and multitasking efficiently on a compact singleboard computer. 
Despite its improved computational power and memory capacity, the Raspberry Pi 4 Model B may encounter limitations when handling extremely demanding tasks or running complex deep learning models in real-time due to its compact form factor and constrained hardware resources. Efficient optimization and task prioritization are crucial for maximizing performance within these constraints. 
#	Operating System of the Pi 
Several Operating Systems were tested (Legacy, Bookworm…), but the best OS for our application was the Bullseye which is a version of the Debian OS. 
#	Camera Module 
To do real-time detection on the Raspberry Pi, a 5MP Raspberry Pi camera was used. 
The 5MP camera module is capable of 1080p video and still image, and it can be connected to Raspberry Pi directly with CSI (Camera Serial Interface). 
The camera must be connected to the CSI port on Raspberry Pi via ribbon cable. 
It is a fixed focus 5MP sensor capable of 2592x1944 stills, but also 1080p30, 720p60 and 640x480p60/90.
# Methodology
- Our methods included converting RGB images from 16 bits to 8 bits and down sampling the resolution from 1080p to 640p. These modifications helped reduce computational load without significantly sacrificing image quality. 
- One of the most impactful strategies involved implementing quantization using OpenVINO, which transformed the model's weights from 32-bit floating point precision to 8-bit floating point precision. This compression technique significantly reduced the model's memory footprint and computational complexity, thereby accelerating inference speed but also keeping the accuracy stable and not lowering it. 
- Additionally, pruning was applied to specific layers of the neural network, eliminating redundant parameters and connections. With a pruning percentage of 20%, unnecessary computational overhead was reduced while preserving the model's essential features for accurate pothole detection. 
- To make the pothole detection model run faster, we moved it to the NCNN framework. This framework is good for devices like the Raspberry Pi because it uses less power. First, we changed the model from a .pt file to a .bin file, going through a step where we turned it into ONNX format. This change made the model work better, especially on devices that don't have a lot of power. 
# Results 
The Raspberry Pi achieved an impressive decrease by 9x in speed from 5.5 seconds to 0.5 seconds inference time, which is considered excellent for an embedded system! However, this comes at the cost of a decreased accuracy but YOLOv8 is robust enough and still achieved a good balance between accuracy and inference time. 
