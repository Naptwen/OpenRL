/*
Copy Right (c) 2022 Useop Gim
Affero GNU GPL v3 license

# Comment
I coded this program for creating an A. I with freedom under positive purpose
for the world. So hope it is useful for mathematicians, programmers, scientists,
and whoever is interested.

# Reference and notation
1. The origin of this software must not be misrepresented; you must not
claim that you wrote the original software. If you use this software
in a product, an acknowledgment in the product documentation would be
appreciated but is not required.
2. Altered source versions must be plainly marked as such,
and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.

# GNU, AFFERO LICENSE
This file is part of the usg_AI Library
you can redistribute it and/or modify it under the terms of the GNU Affero.
Affero General Public License V3 as published by the Free Software
Foundation. <http://www.gnu.org/licenses/> But, WITHOUT ANY WARRANTY without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the GNU General Public License and Affero license for more details

# STD C++ LICENSE
The core part of standard c++ library is base on the using LLVM clang library

# SPECIAL PROVISION
To prevent and make users take responsibility, for the uncontrollable and
unpredictable dangers in the future, a special provision for using this program
for responsibility. This library is restricted about any purpose that breaks the
Laws of Robotic including intentness, negligence, and recklessness also
restricted and users take responsibility.
*/

#include "usg_CLI.hpp"
#if defined(__APPLE__) && defined(__MACH__)
#include "usg_metal.hpp"
#else
#include "usg_Khronos.hpp"
#endif
void test(float* A, float *B, float* C, v_uint sz)
{
	for(int i = 0; i < sz; i++)
		C[i] = A[i] + B[i];
}
int main(int argc, char** argv)
{
	// VMAT<float> A = vecarange<float>(0, 121, 1);
	// int i = 0;
	// openNeural tmp;
	// tmp.load("save/dnn1.json");
	// VMAT<float> trg = {0.2f, 0.8f};
	// printf("start\n");
	// while(i++ < 1000)
	// {
	// 	VMAT<float> out = tmp.forwarding(A);
	// 	printf("->"); printf(out); printf("\n");
	// 	tmp.backWard(out, trg);
	// }
	printf("> [usg_AI] : hello world!\n");
	setting_load();
	if(argc > 1 && argv != nullptr)
	{
		printf("> [usg_AI] : CLI program is running\n");
		usgCLI::menu_check(argc, argv);
	}
	else
	{
		printf("> [usg_AI] : GUI program is running\n");
		usg_AI_GUI_RUN();
	}
	return 0;
}
