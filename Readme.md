## 5.0.1 and 5.0.2 versiona are totally different please reference the new user guide please!

# Quick PPO reinforcement Test
1. Install the program
2. open game.py then place at the left top corner
3. open usg_AI.exe
4. click the RUN GRAPH
5. click the SET then type save/test.txt, 100, result.txt
6. click the pygame window then wait
7. wait 100 times the test iteration (average 10000 times requires but only for test)
8. after fininsh, check the save file _0_result.txt and _1_result.txt file exists.
9. if you want to test for game, click the SET then type save/simul.txt, 100, none
10. click the pygame window then wait

# Q&A

## Why test is so slow?
It is not the algorithm problem it is problem for the extrafn shared library dll file.\
I wrote a very simple shared lib file for just checking and testing small games, so don't use the test dll file for the real problems it is just for testing the program.\
As referencing the form of the extrafn.cpp resource file make your own reward, action, and state functions.\
As giving a tip, most take time part is OpenCV part and Keyboard input part, if you directly get those data from your own program, the speed dramatically increase.

## Only Json file format allowed?
No!, I just wrote the file format as json, it is not required. you can change the extension name anything but plase keep the format.\
Another tip Neural Network weight and bias valuse are base on base64 by Nyffenegger rene.nyffenegger@adp-gmbh.ch.\
Please see his homepage if you have question or interseted in webformating.

## Program unexpectibly shutdown what can I do?
If you don't have idea please open the termina than type 'usg_AI --setting verboseon'\
It shows very detail of the algorithm process. asc font please as it show colored

## On GUI, block doesn't move and can't click!
Press Enter twice.

## Can I use the graphical method for other program?
Yes you can make not only Machine learning algorithm but also any thing that using the FUNC shared library block.\
Only if you keep the format of it, the input and output are working as the same as others.\
Here is the another example for making button click macro by GUI *not machine learning, just a macro program using OpenCV then moving keybord step by step\
![image](https://user-images.githubusercontent.com/47798805/191979230-c539d224-b142-48e9-9cc0-01e32bab6ebf.png)

## Where I can see the format?
Open terminal then type "usg_AI.exe --create [type]" or see wiki page I wrote

# usg_AI v5.0.2 --alpha
![image](https://user-images.githubusercontent.com/47798805/191875624-2a9d79f6-f3ee-4493-865b-07026a5f3a76.png)\
New 5.0.2 version totally different with 5.0.1 version, I am re writing user guide plse see the new version 

## Hot fix, change the console interface display text and shape for node graph for test file (program algorithm doesn't change)
___
<BLOCKQUOTE>
   <summary># PATCH NOTE 4.1.1</summary>
   <PRE>
Sep 23, 2022.
1. New : Change the flow algorithm now you can save all connected nodes.
2. New : Now the model node be the child of the process, and process control all of connected nodes.
3. New : Now all the connected nodes are reference values by chain methods (speed and memory capacity is increase)
Sep 20, 2022.
1. New : Change the algorithm as CFG graph
2. New : Writting user guide. 
3. Fix : Some creating bugs.
4. Edit : Linux working shared library file (not fully checked)
Sep 16, 2022.
1. Fix : Fixing the bug for tree node connections
2. New : If the children are the same generation, the most left child has priority
Sep 15, 2022.
1. New : Seperate functions (state, reward, action) and enviroments
2. New : Coloring the each node by its type
3. New : Change a lot for objective coding
4. New : More easy tree structure
5. New : Multi threading for the running program
Sep 10, 2022.
1. New : Applying shared lib of reward and enviro function for runtime.
2. New : RNN model (dynamical shapes)
3. Feature : operation tree to prepare LSTM
Sep 07, 2022.
1. Edit : Visualizing Algorithms with Enviroments
2. Edit : Change run menu to enviroment and add actions
3. New : loading by tree file which includes cnn, dnn, model, enviro
4. Feature : Looking forward how to make generlizing the reward
Sep 07, 2022.
1. New : Visualizing Algorithms.
2. New : Capture the screen and cropping inside OpenGL (only for winodw now).
3. New : Model running by captured screen.
4. Feature : Learning algorithm applying algorithm and multi thread.
Sep 05, 2022.
1. Edit : Change UI
2. Fix : Font bug fix
3. New : CNN and Model creating
4. Feauture : Visualizing AI creating
Sep 04, 2022.
1. Fix : When display pixel image that were flipped, using matrix transfer fixed it.
2. New : Output text in GUI system by FreeType libraray
3. New : GUI level text input and the object structure of integration GUI interface algorithm.
4. New : File I/O system by GUI for Model algorithm and Neural network algorithm
5. Feature : Tyring to read the video buffer and automatically creating CNN for neural net
Sep 03, 2022.
1. Fix : Edit the PPO algorithm for paraller multiplication
2. New : Construct GUI for OpenGL with GLFW and GLEW
3. New : Reading Screen as Video for using CNN algorithm
4. Feauture : Trying to change OpenGLES for application
</PRE>
___
<!DOCTYPE HTML>
<HTML>

<HEAD>
   USG AI
</HEAD>

<BODY>
   <PRE>
Copy Right (c) 2022 Useop Gim
Affero GNU GPL v3 license
https://github.com/Naptwen/usgAI
<details>
<summary># Comment</summary>
   I coded this program for creating an A. I with freedom under positive purpose
   for the world. So hope it is useful for mathematicians, programmers, scientists,
   and whoever is interested.
</details>
<details>
<summary># Reference and notation</summary>
   1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.
   2. Altered source versions must be plainly marked as such,
   and must not be misrepresented as being the original software.
   3. This notice may not be removed or altered from any source distribution.
</details>
<details>
<summary># GNU, AFFERO LICENSE</summary>
   This file is part of the usg_AI Library
   you can redistribute it and/or modify it under the terms of the GNU Affero.
   Affero General Public License V3 as published by the Free Software
   Foundation. http://www.gnu.org/licenses/ But, WITHOUT ANY WARRANTY without
   even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
   PURPOSE. See the GNU General Public License and Affero license for more details
</details>
<details>
<summary># STD C++ LICENSE</summary>
   The core part of standard c++ library is base on the using LLVM clang library
</details>
<details>
<summary># SPECIAL PROVISION</summary>
   To prevent and make users take responsibility, for the uncontrollable and
   unpredictable dangers in the future, a special provision for using this program
   for responsibility. This library is restricted about any purpose that breaks the
   Laws of Robotic including intentness, negligence, and recklessness also
   restricted and users take responsibility.
</details>
<details>
<summary># REFERENCE FOR BASE 64 SOURCE CODE</summary>

   base64.cpp and base64.h

   base64 encoding and decoding with C++.
   More information at
     <https://renenyffenegger.ch/notes/development/Base64/Encoding-and-decoding-base-64-with-cpp>

   Version: 2.rc.08 (release candidate)

   Copyright (C) 2004-2017, 2020, 2021 René Nyffenegger

   This source code is provided 'as-is', without any express or implied
   warranty. In no event will the author be held liable for any damages
   arising from the use of this software.

   Permission is granted to anyone to use this software for any purpose,
   including commercial applications, and to alter it and redistribute it
   freely, subject to the following restrictions:

   1. The origin of this source code must not be misrepresented; you must not
      claim that you wrote the original source code. If you use this source code
      in a product, an acknowledgment in the product documentation would be
      appreciated but is not required.

   2. Altered source versions must be plainly marked as such, and must not be
      misrepresented as being the original source code.

   3. This notice may not be removed or altered from any source distribution.

   René Nyffenegger rene.nyffenegger@adp-gmbh.ch
</details>
<details>
<summary># Khronos License For OpenGL, OpenCl, OpenCV</summary>
   The core part of OpenCL, OpenGL, OpenCV are following Khronos license
   https://www.khronos.org/legal/Khronos_Apache_2.0_CLA
</details>
</PRE>
</BODY>
</HTML>
