#ifndef _USGNEURAL_H__
#define _USGNEURAL_H__
//Copyright (c) 2022. Useop Gim
//GNU License
#pragma once
#include "usgvmatrix.hpp"
#define NONE 0
#define ADAM 1
#define NADAM 2

//----loss function---------

//must the each sum of X and Y is 1
template<typename K>
vector<K> KL_DIVERGENCE(const vector<K>& X, const vector<K>& Y, bool gradient = false)
{
	assert(X.size() == Y.size());
	if (gradient)
		return X - Y;
	else
		return vecabs(Y * (veclogp1(Y) - veclogp1(X)));
}


template<typename K>
vector<K> MSE2(const vector<K>& X, const vector<K>& Y, bool gradient = false)
{
	if (gradient)
	{
		return X - Y;
	}
	else
	{
		vector<K> T = Y - X;
		return (T * T) * 0.5f;
	}
}

//----activation function---------

template<typename K>
vector<K> linear_x(const vector<K>& A, bool gradient = false)
{
	if (gradient)
	{
		vector<K> C(A.size(), 1);
		return C;
	}
	else
	{
		vector<K> C(A);
		return C;
	}
}


template<typename K>
vector<K> leakReLU(const vector<K>& A, bool gradient = false)
{
	vector<K> C(A);
	if (gradient)
	{
		for (int i = 0; i < A.size(); i++)
			C[i] = (A[i] > 0) ? 1 : 0.3;
	}
	else
	{
		for (int i = 0; i < A.size(); i++)
			C[i] = (A[i] > 0.3 * A[i]) ? A[i] : 0.3 * A[i];
	}
	return C;
}


template<typename K>
vector<K> ReLU(const vector<K>& A, bool gradient = false)
{
	vector<K> C(A);
	if (gradient) 
	{
		for (int i = 0; i < A.size(); i++)
			if (A[i] > 0)
				C[i] = 1;
	}
	else
	{
		for (int i = 0; i < A.size(); i++)
			if (A[i] < 0)
				C[i] = 0;
	}
	return C;
}

//----normalization function---------

template<typename K>
vector<K> znormal(const vector<K>& A, bool gradient = false)
{
	vector<K> C(A);
	if (gradient)
		memset(&C[0], 1, C.size());
	else 
	{
		K _sum = 0;
		K _std = 0;
		for (int i = 0; i < A.size(); i++)
		{
			_sum += A[i];
			_std += pow(A[i], 2);
		}
		K __avg = _sum / A.size();
		_std = _std / (A.size() - 1);

		if(_std == 0)
			memset(&C[0], 0, C.size());
		else 
		{
			for (int i = 0; i < A.size(); i++)
				C[i] = (A[i] - __avg) / (_std);
		}
	}
	return C;
}

template<typename K>
vector<K> softmax(const vector<K>& A, bool gradient = false)
{
	vector<K> C = vecexp(A - vecmax(A));
	K exp_sum = vecsum(C);
	C = C / exp_sum;
	if (gradient)
		return C * (1.0f - C); //1 - 1
	return C;
}


template<typename K>
class openNeural {
private:
	vector<K> z_layer;
	vector<K> x_layer;
	vector<vector<K>(*)(const vector<K>&, bool)> n_layer;
	vector<K> a_layer;
	vector<vector<K>(*)(const vector<K>&, bool)> eq_layer;
	vector<K> vtw_layer;
	vector<K> mtw_layer;
	vector<K> vtb_layer;
	vector<K> mtb_layer;
	vector<K> w_u_layer;
	vector<K> b_u_layer;
	vector<K> gE_layer;
	float gradient_clipping_norm = 0;
	float drop_out_rate = 0.0f;
	float learning_rate = 0.01f;
	int learning_optima = 0;
	int iteration = 0;
	int opt_reset_iteration = 0;
	float beta_1 = 0.9f;
	float beta_2 = 0.999f;
	float zero_preventer = 0.0000001f;
	vector<K>(*loss)(const vector<K>&, const vector<K>&, bool) = MSE2;

	vector<K> __cpu_run() {
		int a_next = 0;
		int w_next = 0;
		int a_shape = 0;
		int w_shape = 0;
		for (int i = 0; i < this->layer_shape.size(); i++)
		{
			a_shape = this->layer_shape[i];
			// X layer update
			vector<K> z_sub(&this->z_layer[a_next], &this->z_layer[a_next + a_shape - 1] + 1);
			vector<K> b_sub(&this->b_layer[a_next], &this->b_layer[a_next + a_shape - 1] + 1);
			memcpy(&this->x_layer[a_next], &(z_sub + b_sub)[0], sizeof(K) * a_shape);
			// Normalization
			vector<K> x_sub(&this->x_layer[a_next], &this->x_layer[a_next + a_shape - 1] + 1);
			vector<K> nx_sub = this->n_layer[i](x_sub, false);
			// Activation
			vector<K> an_sub = this->eq_layer[i](nx_sub, false);
			memcpy(&this->a_layer[a_next], &an_sub[0], sizeof(K) * a_shape);
			// Drop out
			if (this->drop_out_rate != 0)
			{
				//
			}
			//Weight multiplication
			else if (i < this->layer_shape.size() - 1)
			{
				w_shape = this->layer_shape[i] * this->layer_shape[i + 1];
				vector<K> a_sub(&this->a_layer[a_next], &this->a_layer[a_next + a_shape - 1] + 1);
				vector<K> w_sub(&this->w_layer[w_next], &this->w_layer[w_next + w_shape - 1] + 1);
				vector<K> aw = matmul(a_sub, w_sub, 1, this->layer_shape[i], this->layer_shape[i + 1]);
				// For the next shape
				a_next += a_shape;
				w_next += w_shape;
				memcpy(&this->z_layer[a_next], &aw[0], sizeof(K) * this->layer_shape[i + 1]);
			}
		}
		memcpy(&this->output[0], &this->a_layer[a_next], sizeof(K) * a_shape);
		return this->output;
	}
	
	void __cpu_back()
	{
		vector<K> copy_w = this->w_layer;
		vector<K> copy_b = this->b_layer;
		int a_next = this->a_layer.size() - this->layer_shape.back();
		int w_next = this->w_layer.size() - this->layer_shape.back() * this->layer_shape[this->layer_shape.size() - 2];
		int a_shape = 0;
		int w_shape = 0;
		if (this->opt_reset_iteration > 0 && this->iteration % this->opt_reset_iteration == 0)
			this->opt_reset();
		for (int i = this->layer_shape.size() - 1; i > 1; i--)
		{
			try {
				this->iteration += 1;
				a_shape = this->layer_shape[i];
				w_shape = this->layer_shape[i] * this->layer_shape[i - 1];
				vector<K> dE_dA(&this->gE_layer[a_next], &this->gE_layer[a_next + a_shape - 1] + 1);
				if (this->gradient_clipping_norm > 0)
					dE_dA = vecbdd(-gradient_clipping_norm, gradient_clipping_norm, dE_dA);
				vector<K> x_sub(&this->x_layer[a_next], &this->x_layer[a_next + a_shape - 1] + 1);
				vector<K> dA_dNX = this->eq_layer[i](x_sub, true);
				vector<K> dNX_dX = this->n_layer[i](x_sub, true);
				vector<K> dE_dZ = dE_dA * dA_dNX * dNX_dX;
				vector<K> a_sub(&this->a_layer[a_next - this->layer_shape[i - 1]], &this->a_layer[a_next - 1] + 1);
				vector<K> dE_dW = magicmatmul(a_sub, dE_dZ, this->layer_shape[i - 1], this->layer_shape[i]);
				vector<K> w_update;
				vector<K> b_update;
				if (this->learning_optima == ADAM or this->learning_optima == NADAM)
				{
					// weight adam
					vector<K> sub_mw(&this->mtw_layer[w_next], &this->mtw_layer[w_next + w_shape - 1] + 1);
					vector<K> sub_vw(&this->vtw_layer[w_next], &this->vtw_layer[w_next + w_shape - 1] + 1);
					sub_mw = this->beta_1 * sub_mw + (1 - this->beta_1) * dE_dW;
					sub_vw = this->beta_2 * sub_vw + (1 - this->beta_2) * (dE_dW * dE_dW);
					memcpy(&this->mtw_layer[w_next], &sub_mw[0], sizeof(K) * w_shape);
					memcpy(&this->vtw_layer[w_next], &sub_vw[0], sizeof(K) * w_shape);
					vector<K> mdw_corr = sub_mw / K(1 - pow(this->beta_1, this->iteration));
					if (this->learning_optima == NADAM)
						mdw_corr = (this->beta_1 * mdw_corr) + ((1 - this->beta_1) * dE_dW);
					K denominator = (1 - pow(this->beta_2, this->iteration));
					vector<K> vdw_corr = sub_vw / denominator;
					w_update = this->learning_rate * (mdw_corr / (vecsqrt(vdw_corr) + this->zero_preventer));
					// bias adam
					vector<K> sub_mb(&this->mtb_layer[a_next], &this->mtb_layer[a_next + a_shape - 1] + 1);
					vector<K> sub_vb(&this->vtb_layer[a_next], &this->vtb_layer[a_next + a_shape - 1] + 1);
					sub_mb = this->beta_1 * sub_mb + (1 - this->beta_1) * dE_dZ;
					sub_vb = this->beta_2 * sub_vb + (1 - this->beta_2) * (dE_dZ * dE_dZ);
					memcpy(&this->mtb_layer[a_next], &sub_mb[0], sizeof(K) * a_shape);
					memcpy(&this->vtb_layer[a_next], &sub_vb[0], sizeof(K) * a_shape);
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
				vector<K> w_sub(&this->w_layer[w_next], &this->w_layer[w_next + w_shape - 1] + 1);
				vector<K> b_sub(&this->w_layer[a_next], &this->w_layer[a_next + a_shape - 1] + 1);
				w_sub = w_sub - w_update;
				b_sub = b_sub - b_update;
				memcpy(&this->w_layer[w_next], &w_sub[0], sizeof(K) * w_shape);
				memcpy(&this->b_layer[a_next], &b_sub[0], sizeof(K) * a_shape);
				// next gE
				vector<K> gE_sub(&this->gE_layer[a_next - this->layer_shape[i - 1]], &this->gE_layer[a_next - 1] + 1);
				vector<K> next_error = matmul(w_sub, dE_dZ, this->layer_shape[i - 1], this->layer_shape[i], 1);
				memcpy(&this->gE_layer[a_next - this->layer_shape[i - 1]], &gE_sub[0], gE_sub.size());
				// next layer
				a_next -= this->layer_shape[i - 1];
				w_next -= this->layer_shape[i - 2] * this->layer_shape[i - 1];
			}
			catch (int exp)
			{
				printf("Nan, inf catch leaning dismissed");
				this->w_layer = copy_w;
				this->b_layer = copy_b;
				break;
			}
		}
	}

public:
	vector<K> w_layer;
	vector<K> b_layer;
	vector<K> output;
	vector<K> target_val;
	vector<int> layer_shape;
	K error = 100;

	void operator = (const openNeural& B)
	{
		this->w_layer =					B.w_layer;
		this->b_layer =					B.b_layer;
		this->z_layer =					B.z_layer;
		this->x_layer =					B.x_layer;
		this->n_layer =					B.n_layer;
		this->a_layer =					B.a_layer;
		this->eq_layer =				B.eq_layer;
		this->layer_shape =				B.layer_shape;
		this->vtw_layer =				B.vtw_layer;
		this->mtw_layer =				B.mtw_layer;
		this->vtb_layer =				B.vtb_layer;
		this->mtb_layer =				B.mtb_layer;
		this->w_u_layer =				B.w_u_layer;
		this->b_u_layer =				B.b_u_layer;
		this->gE_layer =				B.gE_layer;
		this->output	=				B.output;
		this->gradient_clipping_norm =	B.gradient_clipping_norm;
		this->drop_out_rate =			B.drop_out_rate;
		this->learning_rate =			B.learning_rate;
		this->learning_optima =			B.learning_optima;
		this->iteration =				B.iteration;
		this->opt_reset_iteration =		B.opt_reset_iteration;
		this->beta_1 =					B.beta_1;
		this->beta_2 =					B.beta_2;
		this->zero_preventer =			B.zero_preventer;
		this->loss =					B.loss;
	}

	void add_layer(int layer_size, vector<K>(*active)(const vector<K>&, bool) , vector<K>(*normal)(const vector<K>&, bool) )
	{
		vector<K> zero_vector(layer_size, 0);
		this->b_layer.insert(this->b_layer.end(), zero_vector.begin(), zero_vector.end());
		this->b_u_layer.insert(this->b_u_layer.end(), zero_vector.begin(), zero_vector.end());
		this->z_layer.insert(this->z_layer.end(), zero_vector.begin(), zero_vector.end());
		this->x_layer.insert(this->x_layer.end(), zero_vector.begin(), zero_vector.end());
		this->a_layer.insert(this->a_layer.end(), zero_vector.begin(), zero_vector.end());
		this->vtb_layer.insert(this->vtb_layer.end(), zero_vector.begin(), zero_vector.end());
		this->mtb_layer.insert(this->mtb_layer.end(), zero_vector.begin(), zero_vector.end());
		this->gE_layer.insert(this->gE_layer.end(), zero_vector.begin(), zero_vector.end());
		this->eq_layer.push_back(active);
		this->n_layer.push_back(normal);
		this->layer_shape.push_back(layer_size);
		this->output = zero_vector;
		this->target_val = zero_vector;
	}

	void generate_layer() {
		this->w_layer.clear();
		this->vtw_layer.clear();
		this->mtw_layer.clear();
		this->w_u_layer.clear();
		for (int i = 0; i < this->layer_shape.size() - 1; i++)
		{
			vector<K> zero_vector(this->layer_shape[i] * this->layer_shape[i + 1]);
			this->w_layer.insert(this->w_layer.end(), zero_vector.begin(), zero_vector.end());
			this->vtw_layer.insert(this->vtw_layer.end(), zero_vector.begin(), zero_vector.end());
			this->mtw_layer.insert(this->mtw_layer.end(), zero_vector.begin(), zero_vector.end());
			this->w_u_layer.insert(this->w_u_layer.end(), zero_vector.begin(), zero_vector.end());
		}
	}

	void xavier_init() 
	{
		float n_in = this->layer_shape.front();
		float n_out = this->layer_shape.back();
		float step = n_in + n_out;
		this->w_layer = randvec<K>(-sqrtf(6 / step), sqrtf(6 / step), this->w_layer.size());
	}

	vector<K> run(vector<K> input_val, float drop_out_rate = 0.0f)
	{
		assert(input_val.size() == this->layer_shape[0]);
		assert(drop_out_rate >= 0);
		memcpy(&z_layer[0], &input_val[0], sizeof(K) * input_val.size());
		this->drop_out_rate = drop_out_rate;
		return __cpu_run();
	}

	void learning_set(
		vector<K>(*loss_fn)(const vector<K>&, const vector<K>&, bool), 
		K gradient_clipping_norm = 0.0f, K learning_rate = 0.01f, float drop_out_rate = 0.0f, int opt_reset_iteration = 0, int Learning_optimization = NADAM)
	{
		assert(0 < learning_rate);
		assert(0 <= gradient_clipping_norm);
		this->gradient_clipping_norm = gradient_clipping_norm;
		this->learning_rate = learning_rate;
		this->drop_out_rate = drop_out_rate;
		this->loss = loss_fn;
		this->learning_optima = Learning_optimization;
		this->iteration = 0;
		this->opt_reset_iteration = opt_reset_iteration;
	}
	
	void opt_reset()
	{
		memset(&this->vtw_layer[0], 0, sizeof(K) * this->vtw_layer.size());
		memset(&this->mtw_layer[0], 0, sizeof(K) * this->mtw_layer.size());
		memset(&this->vtb_layer[0], 0, sizeof(K) * this->vtb_layer.size());
		memset(&this->mtb_layer[0], 0, sizeof(K) * this->mtb_layer.size());
		this->iteration = 1;
	}

	void learning_start(vector<K> out_val, vector<K> target_val)
	{
		assert(output.size() == target_val.size());
		assert(this->layer_shape.back() == target_val.size());
		int g_start = this->a_layer.size() - this->layer_shape.back(); 
		this->error = vecsum(this->loss(out_val, target_val, false));
		vector<K> gradient_error = this->loss(out_val, target_val, true); 
		memcpy(&this->gE_layer[g_start], &gradient_error[0], sizeof(K) * gradient_error.size());
		__cpu_back(); 
	}

};

#endif