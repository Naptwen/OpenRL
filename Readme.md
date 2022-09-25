# v 6.0.0
### checking bug now tomorrow release
## PPO
![image](https://user-images.githubusercontent.com/47798805/192142238-bc5ac1b4-665a-4732-884f-13be0755614e.png)
## RNN
![image](https://user-images.githubusercontent.com/47798805/192142292-4779712a-d53c-4216-9a65-7095423b7dbc.png)\
New : back blcock, out block\
New : Order of process flow char is changed gneralized

# Update! v5.0.4!
Hot fix : the dll file vector push_back was wrong algorithm it should have to dist = {} instead dist.push (please fix yourself! or download dll that I fixed)\
New : Now remove algorithm overlapping so the speed of the flow graphc is increase\
New : Now all input and output are pointer chain (speed is increase)

# Upadte! v5.0.3! 
New function for cosst funciton ,(Huber and Cross)\
New Model for reinforcement DDQN D2QN D3QN\
Now, for generlizing the DQN model, doesn't use the discrete action for output for DQN style\
It means all of the action should be continouse output\
In other words, the output layer for the DQN is 1 and for the target DQN has input as sizeof state  + 1 (the addtional input layer size is used for the measuring the action value of output value from the DQN)\
ex) 
- DQN\
Agent Q [inputsize = state size][output size = 1]\
Target Q [inputsize = state size][output size = 1]
- DDQN\
Agent Q [inputsize = state size][output size = 1]\
Target Q [inputsize = state size + 1][output size = 1]
- D2QN\
Agent Q [inputsize = state size][output size = 2]\
Target Q [inputsize = state size][output size = 2]
- D3QN\
Agent Q [inputsize = state size][output size = 2]\
Target Q [inputsize = state size + 1][output size = 2]

# For future V5.1.0!
1. New : RNN! 
2. New : GNN!

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

## Where I can see the format?
Open terminal then type "usg_AI.exe --create [type]" or see user guid for 5.0.2 version

# usg_AI v5.0.2 --alpha
![image](https://user-images.githubusercontent.com/47798805/191979230-c539d224-b142-48e9-9cc0-01e32bab6ebf.png)\
New 5.0.2 version totally different with 5.0.1 version, I am re writing user guide plse see the new version 
5.0.1 version base on the DFS BFS searching child node tree but 5.0.2 version is base on connected hirehcy node\
Connected hirehcy node algorithm save not only children but also all connected parents! \
And 5.0.1 version was based on the copy data for next node but 5.0.2 version directly substitude the data in next node\
Through this algorithm change the speed of prgram is increased!\

## Hot fix, change the console interface display text and shape for node graph for test file (program algorithm doesn't change)
___
<BLOCKQUOTE>
   <summary># PATCH NOTE 4.1.1</summary>
   <PRE>
Sep 24, 2022.
I am preparing for new version v6.x.x it is more focus on providing various algorithm
(if you just want to algorithm please see my past neural network algorithm in python version)
1. New : preparing the v5.1.x version it will includes new cost functions
(Cross Entropy, Binary cross entropy, Relative entropy, RMSE, HUBER, Pesudo Huber)
2. New : New reinforcement model DDQN, D2QN, D3QN, (merged type also), A2C and A3C (using the constructed thread algorithm)
3. New : New RNN (LSTM is created by connection of graph blocks from model RNN)
4. New : Reinforcement learning optimization PER (Prioritized experiance replay buffer)
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
<summary># FreeType FTL license</summary>
      The FreeType Project LICENSE
                    ----------------------------

                            2006-Jan-27

                    Copyright 1996-2002, 2006 by
          David Turner, Robert Wilhelm, and Werner Lemberg



Introduction
============

  The FreeType  Project is distributed in  several archive packages;
  some of them may contain, in addition to the FreeType font engine,
  various tools and  contributions which rely on, or  relate to, the
  FreeType Project.

  This  license applies  to all  files found  in such  packages, and
  which do not  fall under their own explicit  license.  The license
  affects  thus  the  FreeType   font  engine,  the  test  programs,
  documentation and makefiles, at the very least.

  This  license   was  inspired  by  the  BSD,   Artistic,  and  IJG
  (Independent JPEG  Group) licenses, which  all encourage inclusion
  and  use of  free  software in  commercial  and freeware  products
  alike.  As a consequence, its main points are that:

    o We don't promise that this software works. However, we will be
      interested in any kind of bug reports. (`as is' distribution)

    o You can  use this software for whatever you  want, in parts or
      full form, without having to pay us. (`royalty-free' usage)

    o You may not pretend that  you wrote this software.  If you use
      it, or  only parts of it,  in a program,  you must acknowledge
      somewhere  in  your  documentation  that  you  have  used  the
      FreeType code. (`credits')

  We  specifically  permit  and  encourage  the  inclusion  of  this
  software, with  or without modifications,  in commercial products.
  We  disclaim  all warranties  covering  The  FreeType Project  and
  assume no liability related to The FreeType Project.


  Finally,  many  people  asked  us  for  a  preferred  form  for  a
  credit/disclaimer to use in compliance with this license.  We thus
  encourage you to use the following text:

   """
    Portions of this software are copyright © <year> The FreeType
    Project (www.freetype.org).  All rights reserved.
   """

  Please replace <year> with the value from the FreeType version you
  actually use.


Legal Terms
===========

0. Definitions
--------------

  Throughout this license,  the terms `package', `FreeType Project',
  and  `FreeType  archive' refer  to  the  set  of files  originally
  distributed  by the  authors  (David Turner,  Robert Wilhelm,  and
  Werner Lemberg) as the `FreeType Project', be they named as alpha,
  beta or final release.

  `You' refers to  the licensee, or person using  the project, where
  `using' is a generic term including compiling the project's source
  code as  well as linking it  to form a  `program' or `executable'.
  This  program is  referred to  as  `a program  using the  FreeType
  engine'.

  This  license applies  to all  files distributed  in  the original
  FreeType  Project,   including  all  source   code,  binaries  and
  documentation,  unless  otherwise  stated   in  the  file  in  its
  original, unmodified form as  distributed in the original archive.
  If you are  unsure whether or not a particular  file is covered by
  this license, you must contact us to verify this.

  The FreeType  Project is copyright (C) 1996-2000  by David Turner,
  Robert Wilhelm, and Werner Lemberg.  All rights reserved except as
  specified below.

1. No Warranty
--------------

  THE FREETYPE PROJECT  IS PROVIDED `AS IS' WITHOUT  WARRANTY OF ANY
  KIND, EITHER  EXPRESS OR IMPLIED,  INCLUDING, BUT NOT  LIMITED TO,
  WARRANTIES  OF  MERCHANTABILITY   AND  FITNESS  FOR  A  PARTICULAR
  PURPOSE.  IN NO EVENT WILL ANY OF THE AUTHORS OR COPYRIGHT HOLDERS
  BE LIABLE  FOR ANY DAMAGES CAUSED  BY THE USE OR  THE INABILITY TO
  USE, OF THE FREETYPE PROJECT.

2. Redistribution
-----------------

  This  license  grants  a  worldwide, royalty-free,  perpetual  and
  irrevocable right  and license to use,  execute, perform, compile,
  display,  copy,   create  derivative  works   of,  distribute  and
  sublicense the  FreeType Project (in  both source and  object code
  forms)  and  derivative works  thereof  for  any  purpose; and  to
  authorize others  to exercise  some or all  of the  rights granted
  herein, subject to the following conditions:

    o Redistribution of  source code  must retain this  license file
      (`FTL.TXT') unaltered; any  additions, deletions or changes to
      the original  files must be clearly  indicated in accompanying
      documentation.   The  copyright   notices  of  the  unaltered,
      original  files must  be  preserved in  all  copies of  source
      files.

    o Redistribution in binary form must provide a  disclaimer  that
      states  that  the software is based in part of the work of the
      FreeType Team,  in  the  distribution  documentation.  We also
      encourage you to put an URL to the FreeType web page  in  your
      documentation, though this isn't mandatory.

  These conditions  apply to any  software derived from or  based on
  the FreeType Project,  not just the unmodified files.   If you use
  our work, you  must acknowledge us.  However, no  fee need be paid
  to us.

3. Advertising
--------------

  Neither the  FreeType authors and  contributors nor you  shall use
  the name of the  other for commercial, advertising, or promotional
  purposes without specific prior written permission.

  We suggest,  but do not require, that  you use one or  more of the
  following phrases to refer  to this software in your documentation
  or advertising  materials: `FreeType Project',  `FreeType Engine',
  `FreeType library', or `FreeType Distribution'.

  As  you have  not signed  this license,  you are  not  required to
  accept  it.   However,  as  the FreeType  Project  is  copyrighted
  material, only  this license, or  another one contracted  with the
  authors, grants you  the right to use, distribute,  and modify it.
  Therefore,  by  using,  distributing,  or modifying  the  FreeType
  Project, you indicate that you understand and accept all the terms
  of this license.

4. Contacts
-----------

  There are two mailing lists related to FreeType:

    o freetype@nongnu.org

      Discusses general use and applications of FreeType, as well as
      future and  wanted additions to the  library and distribution.
      If  you are looking  for support,  start in  this list  if you
      haven't found anything to help you in the documentation.

    o freetype-devel@nongnu.org

      Discusses bugs,  as well  as engine internals,  design issues,
      specific licenses, porting, etc.

  Our home page can be found at

    https://www.freetype.org


--- end of FTL.TXT ---

</details>
<details>
<summary># Khronos License For OpenGL, OpenCl, OpenCV</summary>
   The core part of OpenCL, OpenGL, OpenCV are following Khronos license
   https://www.khronos.org/legal/Khronos_Apache_2.0_CLA
</details>
</PRE>
</BODY>
</HTML>
