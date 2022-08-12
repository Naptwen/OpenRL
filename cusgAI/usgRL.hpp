#pragma once
#ifndef _USGRL_H__
#define _USGRL_H__
//Copyright (c) 2022. Useop Gim
//GNU License
#include "usgNeural.hpp"
#include <algorithm>
#include <numeric>
#include <deque>
#include <iostream>
/* 
* s : state
* a : action
* r : reward
* ss : next state
* t : termination
*/
template <typename K>
class EXP {
public:
	vector<K> s; // state (t)
	int a; // action
	K r; // reward
	vector<K> ss; //state (t+1)
	bool t; //termination
};

template <typename K>
 class UPDATER {
 public:
	vector<K> Q;
	vector<K> Y;
};

template <typename K>
using reward_fn_type = K(*)(const vector<K>&);

template <typename K>
using enviro_fn_type = bool(*)(const vector<K>&, vector<K>&, vector<K>&, long long);


template <typename K>
class openRL 
{
public:
	reward_fn_type<K> __reward_fn = nullptr;
	enviro_fn_type<K> __enviro_fn = nullptr;
	vector<K> __act_list;
	deque<EXP<K>> replay_buffer;
	vector<openNeural<K>> __agents;
	vector<openNeural<K>> __t_agents;
	int __replay_sz = 32;
	int __mini_sz = 2; 
	int __q_update_fq = 2; 
	int __t_update_fq = 5; 
	float __discount_rate = 0.9f;
	float __update_rate = 0.8f;
	int play_round = 0;

	void RL_ADD_AGENT(const openNeural<K>& Agent)
	{
		this->__agents.push_back(Agent);
		this->__t_agents.push_back(Agent);
	}

	void RL_PLAY_SETTING(const enviro_fn_type<K> envrio_function, const reward_fn_type<K> reward_function, vector<K> act_list)
	{
		this->__reward_fn = reward_function;
		this->__enviro_fn = envrio_function;
		this->__act_list = act_list;
	}

	void RL_REPLAY_SETTING(const int _replay_sz = 32, const  int _mini_sz = 2, const int _q_update_fq = 2, const int _t_update_fq = 5, const float _update_rate = 0.8f)
	{
		this->__replay_sz = _replay_sz;
		this->__mini_sz = _mini_sz;
		this->__q_update_fq = _q_update_fq;
		this->__t_update_fq = _t_update_fq;
		this->__update_rate = _update_rate;
	}

	virtual long long RL_ACTION(const vector<K>& s)=0;

	virtual vector<int> REPLAY_SHUFFLE()=0;

	virtual void RL_LEARN(vector<int>& mini_batch) = 0;

	virtual void RL_TRG_UPDATE() = 0;

	EXP<K>  RL_PLAY(vector<K>& s)
	{
		// play game
		long long act = RL_ACTION(s);
		vector<K> ss(s);
		bool termination = this->__enviro_fn(s, ss, this->__act_list, act);
		EXP<K> exp = { s, act, this->__reward_fn(s), ss, termination };
		// change the state
		memcpy(&s[0], &ss[0], sizeof(float) * ss.size());
		this->replay_buffer.push_back(exp);
		// learning [art
		this->play_round += 1;
		if (this->play_round % this->__q_update_fq == 0 && this->replay_buffer.size() >= this->__replay_sz)
		{
			// buffer update
			while (this->replay_buffer.size() > this->__replay_sz)
				this->replay_buffer.pop_front();
			// replayer re odering
			vector<int> index_list = REPLAY_SHUFFLE();
			// mini batch slicer
			vector<int> mini_batch = vecslicer(index_list, 0, this->__mini_sz);
			RL_LEARN(mini_batch);
			// target softmax-update
			if (this->play_round % this->__t_update_fq == 0) RL_TRG_UPDATE();
		}
		return exp;
	}
	
};
template <typename K>
class openDQN :public openRL<K>
{
	virtual long long RL_ACTION(const vector<K>& s)
	{
		this->__agents[0].run(s);
		long long argmax = 
			max_element(this->__agents[0].output.begin(), 
						this->__agents[0].output.end())
			- this->__agents[0].output.begin();
		return argmax;
	}

	virtual vector<int> REPLAY_SHUFFLE() final
	{
		vector<int> index_list(this->replay_buffer.size());
		iota(index_list.begin(), index_list.end(), 0);
		this->__agents[0];
		shuffle(index_list.begin(), index_list.end(), rngmachine);
		return index_list;
	}

	virtual void RL_LEARN(vector<int> &mini_batch) final
	{
		// mini batch error accumulator
		vector<K> average_Q(this->__agents[0].__neural_layer.layer_shape.back());
		vector<K> average_Y(this->__t_agents[0].__neural_layer.layer_shape.back());
		// mini batch EXP for reinforcment learning
		for (int i = 0; i < mini_batch.size(); i++)
			RL_VALUE(this->replay_buffer[mini_batch[i]], average_Q, average_Y);
		average_Q = average_Q / K(mini_batch.size());
		average_Y = average_Y / K(mini_batch.size());
		this->__agents[0].learning_start(average_Q, average_Y);
	}

	void RL_VALUE(const EXP<K>& replay, vector<K>& mini_Q, vector<K>& mini_Y)
	{
		assert(this->__agents[0].__neural_layer.layer_shape.back()
			== this->__t_agents[0].__neural_layer.layer_shape.back());
		assert(this->__discount_rate > 0);
		float Q = 0;
		float Y = 0;
		if (this->__agents[0].__neural_layer.layer_shape.back() > 1)
		{
			this->__t_agents[0].run(replay.ss);
			Q = this->__agents[0].run(replay.s)[replay.a];
			Y = replay.r + (1.0f - replay.t) * this->__discount_rate * 
				(*max_element(this->__t_agents[0].output.begin(), this->__t_agents[0].output.end()));
			mini_Q[replay.a] += Q;
			mini_Y[replay.a] += Y;
		}
		else
		{
			vector<K> sa(replay.s);
			vector<K> candidates_Y;
			sa.push_back(replay.a);
			for (int i = 0; i < this->__act_list.size(); i++)
			{
				vector<K> ssa(replay.ss);
				sa.push_back(this->__act_list[i]);
				candidates_Y.push_back(this->__t_agents[0].run(ssa)[0]);
			}
			Q = this->__agents[0].run(sa)[0];
			Y = replay.r + (1.0f - replay.t) * this->__discount_rate * (*max_element(candidates_Y.begin(), candidates_Y.end()));
			mini_Q[0] += Q;
			mini_Y[0] += Y;
		}
	}

	virtual void RL_TRG_UPDATE() final
	{
		assert(this->__update_rate > 0.0f);
		if (this->__update_rate != 1.0f)
		{
			this->__t_agents[0].__neural_layer.w_layer =
				this->__update_rate * this->__agents[0].__neural_layer.w_layer
				+ (1 - this->__update_rate) * this->__t_agents[0].__neural_layer.w_layer;
			this->__t_agents[0].__neural_layer.b_layer =
				this->__update_rate * this->__agents[0].__neural_layer.b_layer
				+ (1 - this->__update_rate) * this->__t_agents[0].__neural_layer.b_layer;
		}
		else
		{
			this->__t_agents[0].__neural_layer.w_layer = this->__agents[0].__neural_layer.w_layer;
			this->__t_agents[0].__neural_layer.b_layer = this->__agents[0].__neural_layer.b_layer;
		}
	}
};

template <typename K>
class openPPO :public openRL<K>
{
	virtual long long RL_ACTION(const vector<K>& s)
	{
		this->__agents[0].run(s);
		long long argmax =
			max_element(this->__agents[0].output.begin(),
				this->__agents[0].output.end())
			- this->__agents[0].output.begin();
		return argmax;
	}

	virtual vector<int> REPLAY_SHUFFLE()
	{
		vector<int> index_list(this->replay_buffer.size());
		iota(index_list.begin(), index_list.end(), 0);
		return index_list;
	}

	virtual void RL_LEARN(vector<int>& mini_batch)
	{
		// mini batch EXP for reinforcment learning
		for (int i = 0; i < mini_batch.size(); i++)
			RL_VALUE(this->replay_buffer[mini_batch[i]]);
		for (int i = 0; i < mini_batch.size(); i++)
			this->replay_buffer.pop_front();
	}

	 void RL_VALUE(const EXP<K>& replay)
	{
		assert(this->__discount_rate > 0);
		assert(this->__agents.size() == 2);
		assert(this->__agents[0].__loss_fn == KL_DIVERGENCE);
		K P = 0;
		K Y = 0;
		K epsilon = 0.2;
		//A^(actor network)
		vector<K> candidates = ADVANTAGE_ALGORITHM(replay.ss, this->__act_list);
		vector<K> sa(replay.s);
		sa.push_back(replay.a);
		vector<K> V = this->__agents[1].run(sa);
		vector<K> A_p = { replay.r + this->__discount_rate * V[0] - vecsum(candidates) / candidates.size(), V[1] };
		//policy (critic network)
		vector<K> P_s = this->__agents[0].run(replay.s);
		vector<K> Loss = P_s / this->__t_agents[0].run(replay.s);
		//cliiping
		K clipped_rate = Loss[replay.a];
		if (A_p[0] >= 0) clipped_rate = min(1 + epsilon, clipped_rate);
		else clipped_rate = max(1 - epsilon, clipped_rate);
		Loss[vecargmax(Loss)] = clipped_rate * A_p[0];
		this->__agents[0].learning_start(P_s, Loss); //update critic
		this->__agents[1].learning_start(V , A_p); //update agent
	}

	vector<K> ADVANTAGE_ALGORITHM(const vector<K>& __s, const vector<K>& _act_list)
	{
		vector<K> list_Y_by_a(_act_list.size());
		for (int i = 0; i < _act_list.size(); i++)
		{
			vector<K> __sa(__s);
			__sa.push_back(i);
			list_Y_by_a[i] = this->__agents[1].run(__sa)[0] + this->__agents[1].run(__sa)[1];
		}
		return list_Y_by_a;
	}

	virtual void RL_TRG_UPDATE()
	{
		assert(this->__update_rate > 0.0f);
		if (this->__update_rate != 1.0f)
		{
			this->__t_agents[0].__neural_layer.w_layer =
				this->__update_rate * this->__agents[0].__neural_layer.w_layer
				+ (1 - this->__update_rate) * this->__t_agents[0].__neural_layer.w_layer;
			this->__t_agents[0].__neural_layer.b_layer =
				this->__update_rate * this->__agents[0].__neural_layer.b_layer
				+ (1 - this->__update_rate) * this->__t_agents[0].__neural_layer.b_layer;
			this->__t_agents[1].__neural_layer.w_layer =
				this->__update_rate * this->__agents[1].__neural_layer.w_layer
				+ (1 - this->__update_rate) * this->__t_agents[1].__neural_layer.w_layer;
			this->__t_agents[1].__neural_layer.b_layer =
				this->__update_rate * this->__agents[1].__neural_layer.b_layer
				+ (1 - this->__update_rate) * this->__t_agents[1].__neural_layer.b_layer;
		}
		else
		{
			this->__t_agents[0].__neural_layer.w_layer = this->__agents[0].__neural_layer.w_layer;
			this->__t_agents[0].__neural_layer.b_layer = this->__agents[0].__neural_layer.b_layer;
			this->__t_agents[1].__neural_layer.w_layer = this->__agents[1].__neural_layer.w_layer;
			this->__t_agents[1].__neural_layer.b_layer = this->__agents[1].__neural_layer.b_layer;
		}
	}
};
#endif