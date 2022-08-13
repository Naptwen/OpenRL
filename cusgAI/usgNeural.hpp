#ifndef _USGNEURAL_H__
#define _USGNEURAL_H__
//Copyright (c) 2022. Useop Gim
//GNU License
#pragma once
#include "usgNeural_function.hpp"
#include <fstream>
#define NONE 0
#define ADAM 1
#define NADAM 2


template<typename K>
using neural_function = vector<K>(*)(const vector<K>&, bool);

template<typename K>
using loss_function = vector<K>(*)(const vector<K>&, const vector<K>&, bool);

template<typename K>
class neural_layer {
public:
	vector<K> w_layer;
	vector<K> b_layer;
	vector<K> z_layer;
	vector<K> x_layer;
	vector<neural_function<K>> n_layer;
	vector<K> a_layer;
	vector<neural_function<K>> eq_layer;
	vector<K> vtw_layer;
	vector<K> mtw_layer;
	vector<K> vtb_layer;
	vector<K> mtb_layer;
	vector<K> gE_layer;
	vector<int> layer_shape;

	void deepcopy(const neural_layer& A)
	{
		w_layer= A.w_layer;
		b_layer= A.b_layer;
		z_layer= A.z_layer;
		x_layer= A.x_layer;
		n_layer= A.n_layer;
		a_layer= A.a_layer;
		eq_layer= A.eq_layer;
		vtw_layer= A.vtw_layer;
		mtw_layer= A.mtw_layer;
		vtb_layer= A.vtb_layer;
		mtb_layer= A.mtb_layer;
		gE_layer= A.gE_layer;
		layer_shape= A.layer_shape;
	}
};

template<typename K>
class openNeural {
public:
	neural_layer<K> __neural_layer;
	loss_function<K> __loss_fn;
	vector<K> output;
	vector<K> target_val;
	float gradient_clipping_norm = 0;
	float __drop_out_rate = 0.0f;
	float learning_rate = 0.01f;
	int learning_optima = 0;
	int iteration = 0;
	int opt_reset_iteration = 0;
	float beta_1 = 0.9f;
	float beta_2 = 0.999f;
	float zero_preventer = 0.0000001f;
	K error = 999999999999;

	void file_save(string file)
	{
		ofstream os(file.c_str(), ios::binary);
		int w_sz = this->__neural_layer.w_layer.size();
		int b_sz = this->__neural_layer.b_layer.size();
		os.write((const char*)&w_sz, 4);
		os.write((const char*)&b_sz, 4);
		os.write((const char*)&this->__neural_layer.w_layer[0], w_sz * sizeof(K));
		os.write((const char*)&this->__neural_layer.b_layer[0], b_sz * sizeof(K));
		os.close();
	}

	void file_load(string file)
	{
		ifstream is(file.c_str(), ios::binary);
		int w_sz = 0;
		int b_sz = 0;
		is.read((char*)&w_sz, 4);
		is.read((char*)&b_sz, 4);
		this->__neural_layer.w_layer.resize(w_sz);
		this->__neural_layer.b_layer.resize(b_sz);
		is.read((char*)&this->__neural_layer.w_layer[0], w_sz * sizeof(K));
		is.read((char*)&this->__neural_layer.b_layer[0], b_sz * sizeof(K));
		is.close();
	}

	void deepcopy(const openNeural& B)
	{
		this->__neural_layer.deepcopy(B.__neural_layer);
		this->gradient_clipping_norm =	B.gradient_clipping_norm;
		this->__drop_out_rate =			B.__drop_out_rate;
		this->learning_rate =			B.learning_rate;
		this->learning_optima =			B.learning_optima;
		this->iteration =				B.iteration;
		this->opt_reset_iteration =		B.opt_reset_iteration;
		this->beta_1 =					B.beta_1;
		this->beta_2 =					B.beta_2;
		this->zero_preventer =			B.zero_preventer;
		this->output =					B.output;
		this->target_val =				B.target_val;
		this->__loss_fn =					B.__loss_fn;
	}

	void add_layer(int layer_size, neural_function<K> active, neural_function<K> normal)
	{
		vector<K> zero_vector(layer_size, 0);
		this->__neural_layer.b_layer.insert(this->__neural_layer.b_layer.end(), zero_vector.begin(), zero_vector.end());
		this->__neural_layer.z_layer.insert(this->__neural_layer.z_layer.end(), zero_vector.begin(), zero_vector.end());
		this->__neural_layer.x_layer.insert(this->__neural_layer.x_layer.end(), zero_vector.begin(), zero_vector.end());
		this->__neural_layer.a_layer.insert(this->__neural_layer.a_layer.end(), zero_vector.begin(), zero_vector.end());
		this->__neural_layer.vtb_layer.insert(this->__neural_layer.vtb_layer.end(), zero_vector.begin(), zero_vector.end());
		this->__neural_layer.mtb_layer.insert(this->__neural_layer.mtb_layer.end(), zero_vector.begin(), zero_vector.end());
		this->__neural_layer.gE_layer.insert(this->__neural_layer.gE_layer.end(), zero_vector.begin(), zero_vector.end());
		this->__neural_layer.eq_layer.push_back(active);
		this->__neural_layer.n_layer.push_back(normal);
		this->__neural_layer.layer_shape.push_back(layer_size);
		this->output = zero_vector;
		this->target_val = zero_vector;
	}

	void generate_layer() {
		this->__neural_layer.w_layer.clear();
		this->__neural_layer.vtw_layer.clear();
		this->__neural_layer.mtw_layer.clear();
		for (int i = 0; i < this->__neural_layer.layer_shape.size() - 1; i++)
		{
			vector<K> zero_vector(this->__neural_layer.layer_shape[i] * this->__neural_layer.layer_shape[i + 1]);
			this->__neural_layer.w_layer.insert(this->__neural_layer.w_layer.end(), zero_vector.begin(), zero_vector.end());
			this->__neural_layer.vtw_layer.insert(this->__neural_layer.vtw_layer.end(), zero_vector.begin(), zero_vector.end());
			this->__neural_layer.mtw_layer.insert(this->__neural_layer.mtw_layer.end(), zero_vector.begin(), zero_vector.end());
		}
	}

	void xavier_init() 
	{
		float n_in = this->__neural_layer.layer_shape.front();
		float n_out = this->__neural_layer.layer_shape.back();
		float step = n_in + n_out;
		this->__neural_layer.w_layer = randvec<K>(-sqrtf(6 / step), sqrtf(6 / step), this->__neural_layer.w_layer.size());
	}

	vector<K> run(vector<K> input_val, float drop_out_rate = 0.0f)
	{
		assert(input_val.size() == this->__neural_layer.layer_shape[0]);
		assert(__drop_out_rate >= 0);
		memcpy(&this->__neural_layer.z_layer[0], &input_val[0], sizeof(K) * input_val.size());
		this->__drop_out_rate = __drop_out_rate;
		return __cpu_run();
	}

	void learning_set(loss_function<K > _loss_fn, K _gradient_clipping_norm = 0.0f, K _learning_rate = 0.01f, float _drop_out_rate = 0.0f, int _opt_reset_iteration = 0, int _learning_optimization = NADAM)
	{
		assert(0 < learning_rate);
		assert(0 <= gradient_clipping_norm);
		this->gradient_clipping_norm = _gradient_clipping_norm;
		this->learning_rate = _learning_rate;
		this->__drop_out_rate = _drop_out_rate;
		this->__loss_fn =_loss_fn;
		this->learning_optima = _learning_optimization;
		this->iteration = 0;
		this->opt_reset_iteration = _opt_reset_iteration;
	}
	
	void opt_reset()
	{
		memset(&this->__neural_layer.vtw_layer[0], 0, sizeof(K) * this->__neural_layer.vtw_layer.size());
		memset(&this->__neural_layer.mtw_layer[0], 0, sizeof(K) * this->__neural_layer.mtw_layer.size());
		memset(&this->__neural_layer.vtb_layer[0], 0, sizeof(K) * this->__neural_layer.vtb_layer.size());
		memset(&this->__neural_layer.mtb_layer[0], 0, sizeof(K) * this->__neural_layer.mtb_layer.size());
		this->iteration = 1;
	}

	void learning_start(vector<K> _out_val, vector<K> _target_val)
	{
		assert(output.size() == _target_val.size());
		assert(this->__neural_layer.layer_shape.back() == _target_val.size());
		int g_start = this->__neural_layer.a_layer.size() - this->__neural_layer.layer_shape.back(); 
		this->error = vecsum(this->__loss_fn(_out_val, _target_val, false));
		vector<K> gradient_error = this->__loss_fn(_out_val, _target_val, true); 
		memcpy(&this->__neural_layer.gE_layer[g_start], &gradient_error[0], sizeof(K) * gradient_error.size());
		__cpu_back(); 
	}

	vector<K> __cpu_run() {
		int a_next = 0;
		int w_next = 0;
		int a_shape = 0;
		int w_shape = 0;
		for (int i = 0; i < this->__neural_layer.layer_shape.size(); i++)
		{
			a_shape = this->__neural_layer.layer_shape[i];
			// X layer update
			vector<K> z_sub(&this->__neural_layer.z_layer[a_next], &this->__neural_layer.z_layer[a_next + a_shape - 1] + 1);
			vector<K> b_sub(&this->__neural_layer.b_layer[a_next], &this->__neural_layer.b_layer[a_next + a_shape - 1] + 1);
			memcpy(&this->__neural_layer.x_layer[a_next], &(z_sub + b_sub)[0], sizeof(K) * a_shape);
			// Normalization
			vector<K> x_sub(&this->__neural_layer.x_layer[a_next], &this->__neural_layer.x_layer[a_next + a_shape - 1] + 1);
			vector<K> nx_sub = this->__neural_layer.n_layer[i](x_sub, false);
			// Activation
			vector<K> an_sub = this->__neural_layer.eq_layer[i](nx_sub, false);
			memcpy(&this->__neural_layer.a_layer[a_next], &an_sub[0], sizeof(K) * a_shape);
			// Drop out
			if (this->__drop_out_rate != 0)
			{
				//
			}
			//Weight multiplication
			else if (i < this->__neural_layer.layer_shape.size() - 1)
			{
				w_shape = this->__neural_layer.layer_shape[i] * this->__neural_layer.layer_shape[i + 1];
				vector<K> a_sub(&this->__neural_layer.a_layer[a_next], &this->__neural_layer.a_layer[a_next + a_shape - 1] + 1);
				vector<K> w_sub(&this->__neural_layer.w_layer[w_next], &this->__neural_layer.w_layer[w_next + w_shape - 1] + 1);
				vector<K> aw = matmul(a_sub, w_sub, 1, this->__neural_layer.layer_shape[i], this->__neural_layer.layer_shape[i + 1]);
				// For the next shape
				a_next += a_shape;
				w_next += w_shape;
				memcpy(&this->__neural_layer.z_layer[a_next], &aw[0], sizeof(K) * this->__neural_layer.layer_shape[i + 1]);
			}
		}
		memcpy(&this->output[0], &this->__neural_layer.a_layer[a_next], sizeof(K) * a_shape);
		return this->output;
	}

	void __cpu_back()
	{
		vector<K> copy_w = this->__neural_layer.w_layer;
		vector<K> copy_b = this->__neural_layer.b_layer;
		int a_next = this->__neural_layer.a_layer.size() - this->__neural_layer.layer_shape.back();
		int w_next = this->__neural_layer.w_layer.size() - this->__neural_layer.layer_shape.back() * this->__neural_layer.layer_shape[this->__neural_layer.layer_shape.size() - 2];
		int a_shape = 0;
		int w_shape = 0;
		if (this->opt_reset_iteration > 0 && this->iteration % this->opt_reset_iteration == 0)
			this->opt_reset();
		for (int i = this->__neural_layer.layer_shape.size() - 1; i > 1; i--)
		{
			try {
				this->iteration += 1;
				a_shape = this->__neural_layer.layer_shape[i];
				w_shape = this->__neural_layer.layer_shape[i] * this->__neural_layer.layer_shape[i - 1];
				vector<K> dE_dA(&this->__neural_layer.gE_layer[a_next], &this->__neural_layer.gE_layer[a_next + a_shape - 1] + 1);
				if (this->gradient_clipping_norm > 0)
					dE_dA = vecbdd(-gradient_clipping_norm, gradient_clipping_norm, dE_dA);
				vector<K> x_sub(&this->__neural_layer.x_layer[a_next], &this->__neural_layer.x_layer[a_next + a_shape - 1] + 1);
				vector<K> dA_dNX = this->__neural_layer.eq_layer[i](x_sub, true);
				vector<K> dNX_dX = this->__neural_layer.n_layer[i](x_sub, true);
				vector<K> dE_dZ = dE_dA * dA_dNX * dNX_dX;
				vector<K> a_sub(&this->__neural_layer.a_layer[a_next - this->__neural_layer.layer_shape[i - 1]], &this->__neural_layer.a_layer[a_next - 1] + 1);
				vector<K> dE_dW = magicmatmul(a_sub, dE_dZ, this->__neural_layer.layer_shape[i - 1], this->__neural_layer.layer_shape[i]);
				vector<K> w_update;
				vector<K> b_update;
				if (this->learning_optima == ADAM or this->learning_optima == NADAM)
				{
					// weight adam
					vector<K> sub_mw(&this->__neural_layer.mtw_layer[w_next], &this->__neural_layer.mtw_layer[w_next + w_shape - 1] + 1);
					vector<K> sub_vw(&this->__neural_layer.vtw_layer[w_next], &this->__neural_layer.vtw_layer[w_next + w_shape - 1] + 1);
					sub_mw = this->beta_1 * sub_mw + (1 - this->beta_1) * dE_dW;
					sub_vw = this->beta_2 * sub_vw + (1 - this->beta_2) * (dE_dW * dE_dW);
					memcpy(&this->__neural_layer.mtw_layer[w_next], &sub_mw[0], sizeof(K) * w_shape);
					memcpy(&this->__neural_layer.vtw_layer[w_next], &sub_vw[0], sizeof(K) * w_shape);
					vector<K> mdw_corr = sub_mw / K(1 - pow(this->beta_1, this->iteration));
					if (this->learning_optima == NADAM)
						mdw_corr = (this->beta_1 * mdw_corr) + ((1 - this->beta_1) * dE_dW);
					K denominator = (1 - pow(this->beta_2, this->iteration));
					vector<K> vdw_corr = sub_vw / denominator;
					w_update = this->learning_rate * (mdw_corr / (vecsqrt(vdw_corr) + this->zero_preventer));
					// bias adam
					vector<K> sub_mb(&this->__neural_layer.mtb_layer[a_next], &this->__neural_layer.mtb_layer[a_next + a_shape - 1] + 1);
					vector<K> sub_vb(&this->__neural_layer.vtb_layer[a_next], &this->__neural_layer.vtb_layer[a_next + a_shape - 1] + 1);
					sub_mb = this->beta_1 * sub_mb + (1 - this->beta_1) * dE_dZ;
					sub_vb = this->beta_2 * sub_vb + (1 - this->beta_2) * (dE_dZ * dE_dZ);
					memcpy(&this->__neural_layer.mtb_layer[a_next], &sub_mb[0], sizeof(K) * a_shape);
					memcpy(&this->__neural_layer.vtb_layer[a_next], &sub_vb[0], sizeof(K) * a_shape);
					vector<K> mdb_corr = sub_mb / K(1 - pow(this->beta_1, this->iteration));
					if (this->learning_optima == NADAM)
						mdb_corr = (this->beta_1 * mdb_corr) + ((1 - this->beta_1) * dE_dZ);
					vector<K> vdb_corr = sub_vb / K(1 - pow(this->beta_2, this->iteration));
					b_update = this->learning_rate * (mdb_corr / (vecsqrt(vdb_corr) + this->zero_preventer));
				}
				else
				{
					w_update = this->learning_rate * dE_dW;
					b_update = this->learning_rate * dE_dZ;
				}
				// update 
				vector<K> w_sub(&this->__neural_layer.w_layer[w_next], &this->__neural_layer.w_layer[w_next + w_shape - 1] + 1);
				vector<K> b_sub(&this->__neural_layer.w_layer[a_next], &this->__neural_layer.w_layer[a_next + a_shape - 1] + 1);
				w_sub = w_sub - w_update;
				b_sub = b_sub - b_update;
				memcpy(&this->__neural_layer.w_layer[w_next], &w_sub[0], sizeof(K) * w_shape);
				memcpy(&this->__neural_layer.b_layer[a_next], &b_sub[0], sizeof(K) * a_shape);
				// next gE
				vector<K> gE_sub(&this->__neural_layer.gE_layer[a_next - this->__neural_layer.layer_shape[i - 1]], &this->__neural_layer.gE_layer[a_next - 1] + 1);
				vector<K> next_error = matmul(w_sub, dE_dZ, this->__neural_layer.layer_shape[i - 1], this->__neural_layer.layer_shape[i], 1);
				memcpy(&this->__neural_layer.gE_layer[a_next - this->__neural_layer.layer_shape[i - 1]], &gE_sub[0], gE_sub.size());
				// next layer
				a_next -= this->__neural_layer.layer_shape[i - 1];
				w_next -= this->__neural_layer.layer_shape[i - 2] * this->__neural_layer.layer_shape[i - 1];
			}
			catch (int exp)
			{
				printf("Nan, inf catch leaning dismissed");
				this->__neural_layer.w_layer = copy_w;
				this->__neural_layer.b_layer = copy_b;
				exit(3);
				break;
			}
		}
	}

};

#endif