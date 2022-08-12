GNU AFFERO (C) Useop Gim 2022\
This is c++ version Neural network algorithms\
Test algorithm for DNN with softmax and KLD
```cpp
#include "usgNeural.hpp"
#include <stdio.h>

int main()
{
	openNeural<float> A = openNeural<float>();
	A.add_layer(4, linear_x, linear_x);
	A.add_layer(12, ReLU, znormal);
	A.add_layer(6, ReLU, znormal);
	A.add_layer(3, softmax, linear_x);
	A.generate_layer(); 
	A.opt_reset();
	A.xavier_init();
	vector<float> B = {1,2,3,41};
	vector<float> C = { 0.4f, 0.1f, 0.5f };
	A.learning_set(0, 0.01, 0, KL_DIVERGENCE, 100, NADAM);
	while (A.error >= 0.001)
	{
		vector<float> output = A.run(B);
		A.learning_starat(output, C);
		show_vector(A.output);
		printf("%.5f \n", A.error);
	}
	return 0;
}
```
DQN NO epsilon test
```cpp
#pragma once
#include "usgRL.hpp"
#include <windows.h>

float reward_function(const vector<float>& s)
{
	if (s[0] == s[2] && s[1] == s[3]) return 1;
	else return -((s[0] - s[2]) * (s[0] - s[2]) + (s[1] - s[3]) * (s[1] - s[3]));
}

bool enviro_function(const vector<float>& s, vector<float>& ss, vector<float>& act_list, int a)
{

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dis(0, 4);

	if( a >= 0 && a < 3)
		ss[0] = s[0] + act_list[a];
	else
		ss[1] = s[1] + act_list[a];
	if (ss[0] < 0 || ss[0] > 4)
	{
		ss[0] = 0;
		ss[1] = 0;
		return true;
	}
	if (ss[1] < 0 || ss[1] > 4)
	{
		ss[0] = 0;
		ss[1] = 0;
		return true;
	}
	else if (ss[0] == ss[2] && ss[1] == ss[3])
	{
		ss[2] = dis(gen);
		ss[3] = dis(gen);
		return true;
	}

	return false;
}

void draw(const vector<float>& s)
{
	for (int i = 0; i < 5; i++) 
	{
		for (int j = 0; j < 5; j++)
			if (i == s[0] && j == s[1])
				printf("[x]");
			else if (i == s[2] && j == s[3])
				printf("[G]");
			else
				printf("[ ]");
		printf("\n");
	}
}


int main()
{
	// Neural network 
	openNeural<float> A = openNeural<float>();
	A.add_layer(4, linear_x, linear_x);
	A.add_layer(12, ReLU, znormal);
	A.add_layer(6, ReLU, znormal);
	A.add_layer(6, linear_x, linear_x);
	A.generate_layer(); 
	A.opt_reset();
	A.xavier_init();
	A.learning_set(MSE2);

	openRL R(A, reward_function, enviro_function, {-1,0,1,-1,0,1});
	vector<float> s = { 0,0, 3,3 };
	vector<float> ss = { 0,0, 3, 3 };
	vector<string> action_list = { "up","stop","down","left","stop","right" };
	deque<EXP> exp_list;
	int iter = 0;
	while (iter++ < 10000)
	{
		EXP exp = R.RL_PLAY(s, ss, R.RL_ACTION(s));
		system("cls");
		printf("--------[%d]---------\n",iter);
		draw(exp.s);
		printf("--------[Q]---------\n");
		show_vector(R.Q.run(s));
		printf("--------[s]---------\n");
		show_vector(exp.s);
		printf("--------[a]---------\n");
		printf("act : %s\n", action_list[exp.a].c_str());
		printf("--------[r]---------\n");
		printf("r : %f\n", exp.r);
		printf("--------[ss]---------\n");
		show_vector(exp.ss);
		printf("---------------------\n");
	}
	return 0;
}
```
