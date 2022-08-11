#pragma once
#include "usgNeural.hpp"
#include <algorithm>
/* 
* s : state
* a : action
* r : reward
* ss : next state
* t : termination
*/
struct exp {
	vector<float> s; // state (t)
	int a; // action
	float r; // reward
	vector<float> ss; //state (t+1)
	bool t; //termination
} typedef EXP;

struct updater {
	float Q;
	float Y;
} typedef RL_QY;

RL_QY DQN(EXP replay,
	openNeural<float> &Q, openNeural<float> &TQ, 
	float discount, vector<float> act_list)
{
	Q.run(replay.s);
	TQ.run(replay.ss);
	RL_QY answer;
	answer.Y = replay.r + (1.0f -replay.t) * discount * (*max_element(TQ.output.begin(), TQ.output.end()));
	answer.Q = Q.output[replay.a];
	return answer;
}

void REPLAY_SHUFFLE(vector<EXP> &replay_buffer)
{
	random_device rd;
	default_random_engine rng(rd());
	shuffle(replay_buffer.begin(), replay_buffer.end(), rng);
}

void RL_TRG_UPDATE(openNeural<float>& Q, openNeural<float>& TQ, float update_rate)
{
	assert(update_rate > 0);
	if (update_rate != 1)
	{
		TQ.w_layer = update_rate * Q.w_layer + update_rate * TQ.w_layer;
		TQ.b_layer = update_rate * Q.b_layer + update_rate * TQ.b_layer;
	}
	else 
	{
		TQ.w_layer = Q.w_layer;
		TQ.b_layer = Q.b_layer;
	}
		
}

int RL_ACTION(openNeural<float>& Q, vector<float> s, vector<float> act_list)
{
	Q.run(s);
	int max_index = max_element(Q.output.begin(), Q.output.end()) - Q.output.begin();
	return act_list[max_index];
}
