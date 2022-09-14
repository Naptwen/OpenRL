
How 2 Use v3.0.0

![image](https://user-images.githubusercontent.com/47798805/189315179-2202bd0b-881c-4ec2-a7b9-6f89562606a7.png)

1. Create CNN (the layer types are in https://github.com/Naptwen/usgAI/wiki/usg_AI-3-version)
3. Create DNN (the function types are in https://github.com/Naptwen/usgAI/wiki/usg_AI-3-version)
4. Create Model (the model types are in https://github.com/Naptwen/usgAI/wiki/usg_AI-3-version)
5. Coding reward function and action function then make it as shared library (if you don't know how to do see extrafn.cpp file)

** action function should be formed as c++
```c++ 
void(unsigned int &action)
```
** reward function should be formed as c++
```c++ 
void(std::vector<float> &state, unsigned float &reward, bool &termination)
```

6. Create Enviroments
7. Create Tree\
** The number of connecting CNN to DNN should be match with the channels that you want to extract from the OpenCV\
** 1 CNN means Gray channel, 3 CNN means RGB channel, 4 CNN means RGBA channel, others are not allowed\
Example\
![image](https://user-images.githubusercontent.com/47798805/189314807-ea7ec250-bebd-483f-9d62-8864596d45f8.png)

Above shows 3 channel CNN

9. run it from the command CLI as ./usg_AI.exe --run T [your tree name] [max iteration] [result save file name]
____
Example by given test files
1. Required the tree.txt, enviro.txt, ppo.txt, dnn1.txt, dnn2.txt, cnn1.txt
2. Run the game that I made game.py 
3. Double click usg_AI.exe
4. click the ENVIRO CREATE\
![image](https://user-images.githubusercontent.com/47798805/189294569-09153fad-02a8-4388-8e1f-939ea571651e.png)
5. click the SET ACTION then typing
"extrafn.dll, rewardfn,  actionfn"\
![image](https://user-images.githubusercontent.com/47798805/189294825-21826d04-15de-4b70-9c15-71704fa92825.png)
6. click the Display then cropping the area that you will run the game.py screen\
![image](https://user-images.githubusercontent.com/47798805/189295184-662e2147-f96a-4a34-b4c2-5b53e5e6e72b.png)
7. Save the file name as enviro.txt\
![image](https://user-images.githubusercontent.com/47798805/189295506-adffecb2-5c4d-4f99-a867-17ee9ba00cb0.png)
8. run the game.py\
![image](https://user-images.githubusercontent.com/47798805/189295634-f04103a5-0e75-4c64-8dd9-81d92b7571c4.png)
8. open the command(terminal) Then typing ./usg_AI.exe  --run -t tree.txt 100,000 result.txt\
![image](https://user-images.githubusercontent.com/47798805/189303660-78756217-b02b-49de-9a49-7feb0bb71e7e.png)
9. Wait 100,000 times then the neural network will be saved in result.txt
___
How 2 make Tree
1. only one enviro is allowed, only one model is allowed\
2. click the add\
![image](https://user-images.githubusercontent.com/47798805/189348929-d0ded836-ef3b-45ab-9254-db9c7615ff93.png)
3. Then click empty node then click again the right mouse button\
![image](https://user-images.githubusercontent.com/47798805/189349097-6cadd424-b34f-4259-b499-59f9c3bfefbc.png)
4. Then write the name of file that you already made such as cnn, dnn, model, enviro\
![image](https://user-images.githubusercontent.com/47798805/189349263-cadee30d-909b-4ad6-8246-42e73b09e7d4.png)
5. Then click the block then click the empty space, it will be moved in there\
![image](https://user-images.githubusercontent.com/47798805/189349369-c857972e-58a6-48d8-af54-aeed030aa9d6.png)
6. Add another\
![image](https://user-images.githubusercontent.com/47798805/189349417-fdff35b8-bbce-4cc1-bed9-dfe02a361f69.png)
7. click the node that you made then click another node\
![image](https://user-images.githubusercontent.com/47798805/189349545-c539919e-c730-45b2-b2cd-10526df11dd2.png)
8. You can see the connection from empty to test.txt the direction is from empty to test.txt\
![image](https://user-images.githubusercontent.com/47798805/189349638-72551619-0c4f-4676-8a56-0d15964bbddc.png)
9. Make shape that you want\
![image](https://user-images.githubusercontent.com/47798805/189349731-d60d7e87-a454-44c2-88bb-7e0f2812b13d.png)\
![image](https://user-images.githubusercontent.com/47798805/189349799-0f677458-f5d7-485f-9fda-bc467cb2a003.png)
10. If you want to disconnect the node direction double click the node that you want to disconnect\
![image](https://user-images.githubusercontent.com/47798805/189349901-ba52b86a-3486-4cd5-8f65-a951861670ee.png)
11. If you want to delete the node click the node then press the Delete key\
![image](https://user-images.githubusercontent.com/47798805/189349988-d71c1e17-672e-4164-b25a-74757ac92366.png)
12. If you want to save the node click the node then click the save button\
![image](https://user-images.githubusercontent.com/47798805/189350105-d46bfff8-c142-46bd-8f24-e66797110905.png)
