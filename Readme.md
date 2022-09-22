#usg_AI v5.1.0 --alpha
![image](https://user-images.githubusercontent.com/47798805/191849321-5218f7f2-2ff4-4d5c-993e-2480e86c5b56.png)
* New : now the all connected blocks are save, load and running
* New : for connected dynamical graph, many algorihtms are changed
* Fix : some bug 

# usg_AI v5.0.2 --alpha
* Remove : Now the reward, state active block are removed.
* New : The shared library is now more generlized, you can use the visual graph flow control for other program.


# usg_AI v5.0.1 --alpah
![image](https://user-images.githubusercontent.com/47798805/191221253-0410825b-0c3f-496a-95f5-956bc43d0f5c.png)
* New : Change algorithms 
* New : Change from tree structure to Direction graph CFG
* New : Change almost of all code structure and GUI menu also

I removed the other GUI, only left making CFG graph and running the program\
because the GUI system is unnecessary if you use the text edit file and it is enough to create various models or just use the console interface.\
I totally changed a lot of the code and algorithm to make the program more general and increase the speed.\
The most important part is to change the tree structure to graph flow graph CFG.\
The most taking time part was the shared library parts.\
Thus if you really want to apply for the program, don't use the example shared library (my example are just super simple code parts).

# usg_AI V4.1.0 --alpha
EXAMPLE FOR TEST FILE
![test](https://user-images.githubusercontent.com/47798805/190431307-d868c26a-527d-422d-b327-13a777429779.gif)

Requirement\
Above OpenCV 2.0 , Above OpenGL 3.0, Above OpenCL 2.0, Above C++ 11, Window 64x
___
<BLOCKQUOTE>
   <summary># PATCH NOTE 4.1.1</summary>
   <PRE>
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
