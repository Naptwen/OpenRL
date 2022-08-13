GNU AFFERO (C) Useop Gim 2022\
This is c++ version Neural network algorithms\
This is test algorithm for PPO with sharp reward policy
```cpp
#pragma once
#include "usgRL.hpp"
#include <windows.h>

float reward_function(const vector<float>& s)
{
	if (s[0] == s[2] && s[1] == s[3]) return 1;
	else return -sqrtf(((s[0] - s[2]) * (s[0] - s[2]) + (s[1] - s[3]) * (s[1] - s[3])));
}

bool enviro_function(const vector<float>& s, vector<float>& ss, vector<float>& act_list, long long a)
{
	uniform_int_distribution<int> dis(0, 4);
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
		ss[2] = dis(rngmachine);
		ss[3] = dis(rngmachine);
		if (ss[2] == 0 && ss[3] == 0)
		{
			uniform_int_distribution<int> dis2(1, 4);
			ss[2] = dis2(rngmachine);
			ss[3] = dis2(rngmachine);
		}
		return false;
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
	auto A = openNeural<float>();
	A.add_layer(4, linear_x, linear_x);
	A.add_layer(12, ReLU, linear_x);
	A.add_layer(6, ReLU, linear_x);
	A.add_layer(5, softmax, linear_x);
	A.generate_layer();
	A.opt_reset();
	A.xavier_init();
	A.learning_set(KL_DIVERGENCE, 0,0.03f, 0.0, 100, NADAM);

	auto B = openNeural<float>();
	B.add_layer(5, linear_x, linear_x);
	B.add_layer(36, leakReLU, znormal);
	B.add_layer(1, linear_x, linear_x);
	B.generate_layer();
	B.opt_reset();
	B.xavier_init();
	B.learning_set(MSE2, 0, 0.03f, 0.0, 100, NADAM);

	auto R = openPPO<float>();
	R.RL_ADD_AGENT(A);
	R.RL_ADD_AGENT(B);
	R.RL_REPLAY_SETTING(32, 4, 3, 5, 0.01f);
	R.RL_PLAY_SETTING(enviro_function, reward_function,  { -1,0,1,-1, 1 });

	vector<float> s = { 0,0, 3,3 };
	vector<float> ss = { 0,0, 3, 3 };
	vector<string> action_list = { "up","    stop","  down","  left","  right" };
	int iter = 0;
	while (iter++ < 200)
	{
		system("cls");
		EXP<float> exp = R.RL_PLAY(s);
		printf("--------[%d]---------\n",iter);
		draw(exp.s);
		printf("--------[QV]---------\n");
		vector<float> sa = veclineup(s, {float(exp.a) });
		printf("--------[p]---------\n");
		show_vector(action_list);
		show_vector(R.__agents[0].run(exp.s));
		if(R.__agents.size() > 1)
			show_vector(R.Q_S_W_A(R.__agents[1], exp.s, R.__act_list));
		if(R.LOSS_X.size() > 0)
			show_vector(R.LOSS_X), show_vector(R.LOSS_Y);
		printf("--------[s]---------\n");
		show_vector<float>(exp.s);
		printf("--------[a]---------\n");
		printf("act : %s\n", action_list[exp.a].c_str());
		printf("--------[r]---------\n");
		printf("r : %f\n", exp.r);
		printf("--------[ss]---------\n");
		show_vector<float>(exp.ss);
		printf("---------------------\n");
	}

	return 0;
}
```
