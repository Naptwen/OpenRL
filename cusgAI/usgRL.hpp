#pragma once
#ifndef _USGRL_H__
#define _USGRL_H__
//Copyright (c) 2022. Useop Gim
//GNU License
#include "usgNeural.hpp"
#include <algorithm>
#include <numeric>
#include <deque>
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
	vector<float> Q;
	vector<float> Y;
} typedef UPDATER;

void DQN(UPDATER &updater, const EXP &replay, openNeural<float> &agent, openNeural<float> &target_agent, vector<float> act_list, float discount = 0.8f)
{
	assert(agent.layer_shape.back() == target_agent.layer_shape.back());
	assert(discount > 0);
	float Q = 0;
	float Y = 0;
	if (agent.layer_shape.back() > 1)
	{
		target_agent.run(replay.ss); 
		Q = agent.run(replay.s)[replay.a]; 
		Y = replay.r + (1.0f - replay.t) * discount * (*max_element(target_agent.output.begin(), target_agent.output.end()));
	}
	else
	{
		vector<float> sa(replay.s);
		vector<float> candidates_Y;
		sa.push_back(replay.a);
		for (int i = 0; i < act_list.size(); i++) 
		{
			vector<float> ssa(replay.ss);
			sa.push_back(act_list[i]);
			candidates_Y.push_back( target_agent.run(ssa)[0]);
		}
		Q = agent.run(sa)[0];
		Y = replay.r + (1.0f - replay.t) * discount * (*max_element(candidates_Y.begin(), candidates_Y.end()));
	}
	updater.Q = vector<float>(agent.layer_shape.back());
	updater.Y = vector<float>(target_agent.layer_shape.back());
	updater.Q[replay.a] += Q;
	updater.Y[replay.a] += Y;
}

vector<int> REPLAY_SHUFFLE(const deque<EXP> &replay_buffer)
{
	random_device rd;
	default_random_engine rng(rd());
	vector<int> index_list(replay_buffer.size());
	iota(index_list.begin(), index_list.end(), 0);
	shuffle(index_list.begin(), index_list.end(), rng);
	return index_list;
}

void RL_TRG_UPDATE(openNeural<float>& Q, openNeural<float>& TQ, float update_rate = 0.05f)
{
	assert(update_rate > 0);
	if (update_rate != 1)
	{
		TQ.w_layer = update_rate * Q.w_layer + (1 - update_rate) * TQ.w_layer;
		TQ.b_layer = update_rate * Q.b_layer + (1 - update_rate) * TQ.b_layer;
	}
	else 
	{
		TQ.w_layer = Q.w_layer;
		TQ.b_layer = Q.b_layer;
	}
}

class openRL 
{
	vector<float> act_list;
	openNeural<float> TQ;
	int replay_sz = 32;
	int mini_sz = 2; 
	int q_update_fq = 2; 
	int t_update_fq = 5; 
	float update_rate = 0.8f;
	float(*reward_fn)(const vector<float>&) = NULL;
	bool(*enviro_fn)(const vector<float>&, vector<float>&, vector<float>&, int) = NULL;
	deque<EXP> replay_buffer;
public:
	openNeural<float> Q;
	int play_round = 0;
	openRL(
		openNeural<float>& agent_neural,
		float(*reward_fn)(const vector<float>&),
		bool(*enviro_fn)(const vector<float>&, vector<float>&, vector<float>&, int),
		vector<float> act_list)
	{
		this->Q = agent_neural;
		this->TQ = agent_neural;
		this->reward_fn = reward_fn;
		this->enviro_fn = enviro_fn;
		this->act_list = act_list;
	}

	void RL_SETTING(
		int replay_sz = 32, int mini_sz = 2, 
		int q_update_fq = 2, int t_update_fq = 5, float update_rate = 0.8f)
	{
		this->replay_sz = replay_sz;
		this->mini_sz = mini_sz;
		this->q_update_fq = q_update_fq;
		this->t_update_fq = t_update_fq;
		this->update_rate = update_rate;
	}

	int RL_ACTION(vector<float> s)
	{
		this->Q.run(s);
		int argmax = max_element(Q.output.begin(), Q.output.end()) - Q.output.begin();
		return argmax;
	}

	EXP RL_PLAY(vector<float>& s, vector<float>& ss, const int action)
	{
		
		// play game
		bool termination = RL_ENVIRO(s, ss, action);
		EXP exp = { s, action, RL_REWARD(s), ss, termination };
		// change the state
		memcpy(&s[0], &ss[0], sizeof(float) * ss.size());
		this->replay_buffer.push_back(exp);
		// learning [art
		this->play_round += 1;
		// buffer update
		while (this->replay_buffer.size() > this->replay_sz)
		{
			this->replay_buffer.pop_front();
		}
		if (this->play_round % q_update_fq == 0 && this->replay_buffer.size() == this->replay_sz)
		{
			// replayer re odering
			vector<int> index_list = REPLAY_SHUFFLE(this->replay_buffer); 
			// mini batch slicer
			vector<int> mini_batch = vecslicer(index_list, 0, this->mini_sz); 
			// mini batch error accumulator
			UPDATER updater;
			updater.Q = vector<float>(this->Q.layer_shape.back()); 
			updater.Y = vector<float>(this->TQ.layer_shape.back());
			// mini batch exp for reinforcment learning
			for (int i = 0; i < mini_batch.size(); i++)
				DQN(updater, this->replay_buffer[mini_batch[i]], this->Q, this->TQ, this->act_list);
			Q.learning_start(updater.Q, updater.Y);
			// target softmax-update
			if (this->play_round % this->t_update_fq == 0) RL_TRG_UPDATE(this->Q, this->TQ, this->update_rate);
		}
		return exp;
	}

	float RL_REWARD(const vector<float>& s)
	{
		return this->reward_fn(s);
	}

	bool RL_ENVIRO(const vector<float> &s, vector<float> &ss, const int action)
	{
		return this->enviro_fn(s, ss, this->act_list, action);
	}


};


#endif