# How 2 Use
![image](https://user-images.githubusercontent.com/47798805/190436865-5213804b-73ab-42c1-8b3a-ac9076e09ce9.png)
* ```CNN CREATE```: It is for making CNN layer
* ```NEURAL CREATE```: It is for making DNN layer
* ```MODEL CREATE```: It is for making Reinforcement Learning model layer
* ```ENVIRO CRREATE```: It is for cropping the display area and loading shared libs
* ```VISUAL CREATE```: It is for creating visualization tree constructure 

## 1. Create CNN
![image](https://user-images.githubusercontent.com/47798805/190437471-860010bc-8619-478b-aa30-4b0fd9e3dcda.png)
* ```HEIGHT SIZE```: It is for height size for CNN layer
* ```NEURAL CREATE```: It is for width size for CNN layer
* ```LAYER TYPE```: It is for CNN layer models 
* ![image](https://user-images.githubusercontent.com/47798805/190437302-ab5af6c1-0b37-4e7d-a59b-62955ebc8bbe.png)
* ```PUSH```: It is for add layer in CNN
* ```POP```: It is for delete the last layer from the CNN
## 2. Create DNN
![image](https://user-images.githubusercontent.com/47798805/190438261-81dfc32a-d7e7-4a66-9c01-dec4fb4d5fec.png)
* ```LAYER INPUT```: It is for setting the single layer size
* ```LAYER ACT F```: It is for setting activation function for the single layer
* ```LAYER NOR F```: It is for setting batch normalization function for the single layer
* ![image](https://user-images.githubusercontent.com/47798805/190438901-30d899ff-406b-4c8f-82d0-2ce13cb7e0e1.png)
* ```PUSH```: It is for add layer into the dnn layer
* ```POP```: It is for delete the last layer from the dnn layer
* ```LOSS F```:It is for setting the loss(cost) function for the dnn layer
* ![image](https://user-images.githubusercontent.com/47798805/190439656-ca2a5fbf-3f43-41a3-964a-29730f61c3e2.png)
* ```LEARN RATE```: It is for setting learning rate for the dnn layer
* ```DROP RATE```: It is for setting drop rate for the dnn layer
* ```OPT RESET```: It is for setting the optimization reset frequency for the dnn layer
## 3. Create Model
![image](https://user-images.githubusercontent.com/47798805/190440292-e84f6dfc-d3d3-4709-889b-c4a529dade9f.png)
* ```REPLAY SIZE``` : It is for setting the replay buffer size
* ```MINI SIZE```: It is for setting the mini batch size
* ```UPDATE FREQUENCY```: It is for setting the agent update frequency 
* ```TARGET UPDATE```: It is for setting the target agent update frequency
* ```DICCOUNT RATE```: It is for the discount rate
* ```TRG UP RATE```: It is for target agent update rate (soft update rate)
* ```MODEL TYPE```: It is for setting the reinforcement model type
* ![image](https://user-images.githubusercontent.com/47798805/190441134-153bb629-842a-4727-b769-8c3947ef9d03.png)
## 4. Create Enviroments
![image](https://user-images.githubusercontent.com/47798805/190441288-c26545ba-c0c1-49bc-953d-943b0e59d876.png)
* ```DISPLAY``` : It is for showing the screen to crop the scrreen, it must be on
* ```CROP``` : It is for cropping the screen
* ```SHARE LIB``` : It is for setting shared library, [PATH],[STATE],[REWARD],[ACTION] with comma seperate
* ```SAVE xxx``` : It is for save the each shared library functions
## 5. Create Tree
![image](https://user-images.githubusercontent.com/47798805/190232413-d561f63d-c8ca-4916-ac33-563461c5ad5d.png)

* ```Add``` : add a new node
![image](https://user-images.githubusercontent.com/47798805/190232934-965f71bb-0226-4f27-8398-79992f0dde0f.png)

* ```Mouse right click``` : Edit the name of the node (if the file name is exist the color of the node is automatically changed)
![image](https://user-images.githubusercontent.com/47798805/190232978-ac57445c-8449-4433-b00b-b23c8493e939.png)
![image](https://user-images.githubusercontent.com/47798805/190233034-055c513d-43a4-4a25-b4f6-a85bc9ac7e7e.png)
### If the name of the node is p or x then it is automatically changed as the operation block + and x
![image](https://user-images.githubusercontent.com/47798805/190233707-280f3656-7bdb-4128-8f4a-22851b03216d.png)
- operation block means the output of operation block is the smae as all input values by operations + or x

* ```Mouse left click``` : move the location of clicked node or connect the node from clicked node to the last clicked node
![image](https://user-images.githubusercontent.com/47798805/190233183-9b5fc21d-ee81-4852-a9dd-93e8f2aedf9a.png)
![image](https://user-images.githubusercontent.com/47798805/190233330-22a94dc7-bbdd-4560-9b2a-6d571d197b2e.png)

* ```Mouse left double click``` : disconnect all of its connection
![image](https://user-images.githubusercontent.com/47798805/190234104-870381ec-f87e-4c7f-9524-02e690044069.png)
![image](https://user-images.githubusercontent.com/47798805/190234132-7b0b547c-5d7b-4703-98ce-47d17bf420a0.png)

* ```Delete key```: delete the block
![image](https://user-images.githubusercontent.com/47798805/190234189-57dbe3a7-6878-47cb-a3cd-4ba70635a3fc.png)

*```LOAD```: load tree file

## RUN TREE
![image](https://user-images.githubusercontent.com/47798805/190443352-11dbd7b2-4f76-454e-b4ba-cfb602d19a22.png)
*```SET```: It is for setting the running information, [tree path],[max iteration],[save path] (cooma seperate)
*```RUN```: It is for running the RL model. If the multi of setting file is true, multiple model can be running

## SETTING FILE
* The file name must "usg_AI_setting.txt"
* ![image](https://user-images.githubusercontent.com/47798805/190444254-b3fc4171-fa2b-40f8-8216-c62acc5bf76c.png)
* ```show``` : It is not working for version 4.x.x (it will be used for neural network working visualization)
* ```multi```: It is for multiple thread running model
* ```gpgpu```: It is for gpgpu (openCL) version calculation
* ```network```: It is not working for version 4.x.x (it will be used for the TCP/IP multiple socket network)
* ```verbose```: It is for showing detail of running the program

## Console Interface
* Using terminal then run with ```--help``` option
___
# BUG
1. Wrong load file name will make bug
2. Wrong file type will make bug
3. If can't click anything, ENTER the Keyboard
4. Wrong format of tree design, especially for ppo, dqn, sac, makes bug and fatal error
5. When disconnect the node somtimes it shows un expeected runtime error please resatart the program
6. For basic RL model the enviroments must have 
    1. one operation child node, 
    2. one action function
    3. one reward function
    4. one model child node
