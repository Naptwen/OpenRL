#ifndef _USGNEURAL_FN_H__
#define _USGNEURAL_FN_H__
//Copyright (c) 2022. Useop Gim
//GNU License
#pragma once
#include "usgvmatrix.hpp"

//----__loss function---------

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
		return X - Y;
	else
		return (Y - X) * (Y - X) * 0.5f;
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

		if (_std == 0)
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
vector<K> min_max_normal(const vector<K>& A, bool gradient = false)
{
	vector<K> C(A);
	if (gradient)
		memset(&C[0], 1, C.size());
	else
	{
		K max_val = vecmax(A);
		K min_val = vecmin(A);
		if(max_val == min_val)
			memset(&C[0], 1, C.size());
		else
			for (int i = 0; i < A.size(); i++)
				C[i] = A[i] - min_val / (max_val - min_val);
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
#endif