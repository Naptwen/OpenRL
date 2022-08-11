#ifndef _VMATRIX_H__
#define _VMATRIX_H__
//Copyright (c) 2022. Useop Gim
//GNU License
#pragma once
#include <vector>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <cassert>
#include <random>

using namespace std;
#define dot >>
#define cross %
#define THE_FIRST_VEC(X) X[0]
#define THE_FIRST(X) X[0][0]
#define SIGN(X) (X<0)?-1:1
#define SHRINK(X) vec2d2vec1d(X)
#define CLUMPLE &
struct matrixT {};
const matrixT T{}; //transpose
struct matrixP {};
const matrixP p{}; //pesudo inverse

//---------code-----------------------------

template<typename K>
void show_vector(const vector<K>& A) {
    for (auto vec : A)
        printf("%.3f,  ", vec);
    printf("\n");
}
template<>
void show_vector(const vector<int>& A) {
    for (auto vec : A)
        printf("%2d  ", vec);
    printf("\n");
}
template<>
void show_vector(const vector<string>& A) {
    for (auto vec : A)
        printf("%s  ", vec.c_str());
    printf("\n");
}
template<typename K>
void show_vector(const vector<vector<K>>& A) {
    for (auto vec : A) {
        for (auto v : vec) {
            printf("%.2f,  ", v);
        }
        printf("\n");
    }
}
template<typename K>
void show_size(const vector<vector<K>>& A) {
    printf("%lu x %lu\n", A.size(), A[0].size());
}

//---------------algorithm--------------------

//if A[i] > maximum then true else false
template<typename K>
vector<bool> biggermax(const vector<K> &A, const K maximum)
{
    vector<bool> bool_list(A.size());
    for (int i = 0; i < A.size(); i++)
        if (A[i] >= maximum)
            bool_list[i] = true;
        else
            bool_list[i] = false;
    return bool_list;
}
//if A[i] > B[i] then true else false
template<typename K>
vector<bool> biggervec(const vector<K> &A, const vector<K> B)
{
    assert(A.size() == B.size());
    vector<bool> bool_list(A.size());
    for (int i = 0; i < A.size(); i++)
        if (A[i] >= B[i])
            bool_list[i] = true;
        else
            bool_list[i] = false;
    return bool_list;
}

//give random size vector
template<typename K>
vector<K> randvec(float min, float max, long long int sz)
{
    assert(min <= max);
    vector<K> v(sz, 0);
    random_device rnd_device;
    mt19937 mersenne_engine{ rnd_device() };
    uniform_real_distribution<float> dist{ min, max };
    auto gen = [&dist, &mersenne_engine](){return dist(mersenne_engine);};
    generate(v.begin(), v.end(), gen);
    return v;
}

//limited vector
template<typename K>
vector<K> vecbdd(float min, float max, const vector<K> &A)
{
    vector<K> B(A);
    for (int i = 0; i < B.size(); i++)
        B[i] = (B[i] > max) ? max : (B[i] < min) ? min : B[i];
    return B;
}

// taking sqrt for vector
template<typename K>
vector<K> vecsqrt(const vector<K>& A) {
    vector<K> C(A);
    for (int i = 0; i < A.size(); i++)
    {
        assert(A[i] >= 0);
        C[i] = sqrt(A[i]);
    }
    return C;
}

// taking exp for vector + 1
template<typename K>
vector<K> vecexp(const vector<K>& A) {
    vector<K> C(A);
    for (int i = 0; i < A.size(); i++)
        C[i] = exp(A[i]);
    return C;
}

// taking log for vector
template<typename K>
vector<K> veclog(const vector<K>& A) {
    vector<K> C(A);
    for (int i = 0; i < A.size(); i++)
        C[i] = log(A[i]);
    return C;
}

// taking log for vector + 1
template<typename K>
vector<K> veclogp1(const vector<K>& A) {
    vector<K> C(A);
    for (int i = 0; i < A.size(); i++)
        C[i] = log(A[i] + 1);
    return C;
}


//max
template<typename K>
K vecmax(const vector<K>& A)
{
    K max = A[0];
    for (int i = 1; i < A.size(); i++)
        max = (A[i] > max) ? A[i] : max;
    return max;
}

//max
template<typename K>
K vecsum(const vector<K>& A)
{
    K sum = 0;
    for (int i = 0; i < A.size(); i++) sum += A[i];
    return sum;
}

//max int
template<typename K>
int vecmaxindex(const vector<K>& A)
{
    K max = A[0];
    int index = 0;
    for (int i = 1; i < A.size(); i++) 
    {
        max = (A[i] > max) ? A[i] : max;
        index = i;
    }
    return index;
}


//---------1d vector operator-----------

template<typename K>
K operator dot (const vector<K>& AT, const vector<K>& B) {
    if (AT.size() != B.size()) {
        show_size(AT);
        throw length_error("[vector dot] the size of vectors is not match");
    }
    K ans = 0;
    for (int i = 0; i < AT.size(); i++)
        ans += AT[i] * B[i];
    return ans;
}
template<typename K>
vector<K> operator cross (const vector<K>& A, const vector<K>& B) {
    if (A.size() != 3 || B.size() != 3) {
        show_size(A);
        show_size(B);
        throw length_error("[vector cross] the size of vectors is not 3d space");
    }
    vector<K> C =
    {
        (A[1] * B[2] - A[2] * B[1]),
        -(A[0] * B[2] - A[2] * B[0]),
        (A[0] * B[1] - A[1] * B[0])
    };
    return C;
}
template<typename K>
vector<K> operator + (const vector<K>& A, const vector<K>& B) {
    if (A.size() != B.size())
    {
        throw length_error("[vector operator +] the size of vectors is not match");
    }
    vector<K> C(A.size());
#pragma omp for
    for (int i = 0; i < A.size(); i++)
        C[i] = A[i] + B[i];
    return C;
}
template<typename K>
vector<K> operator + (const K& A, const vector<K>& B) {
    vector<K> C(B.size());
#pragma omp for
    for (int i = 0; i < A.size(); i++)
        C[i] = A + B[i];
    return C;
}
template<typename K>
vector<K> operator + (const vector<K>& B, const K& A) {
    vector<K> C(B.size());
#pragma omp for
    for (int i = 0; i < B.size(); i++)
        C[i] = A + B[i];
    return C;
}
template<typename K>
vector<K> operator - (const vector<K>& A, const vector<K>& B) {
    if (A.size() != B.size()) {
        throw length_error("[vector opertator] - :  the size of vectors is not match");
    }
    vector<K> C(A.size());
#pragma omp for
    for (int i = 0; i < A.size(); i++)
        C[i] = A[i] - B[i];
    return C;
}
template<typename K>
vector<K> operator - (const vector<K>& A) {
    vector<K> C(A.size());
#pragma omp for
    for (int i = 0; i < A.size(); i++)
        C[i] = -A[i];
    return C;
}
template<typename K>
vector<K> operator - (const K& B, const vector<K>& A) {
    vector<K> C(A);
#pragma omp for
    for (int i = 0; i < A.size(); i++)
        C[i] = B - C[i];
    return C;
}
template<typename K>
vector<K> operator - (const vector<K>& A, const K& B) {
    vector<K> C(A);
#pragma omp for
    for (int i = 0; i < A.size(); i++)
        C[i] = C[i] - B;
    return C;
}
template<typename K>
vector<K> operator * (const K& c, const vector<K>& A) {
    vector<K> ans(A.size());
#pragma omp for
    for (int i = 0; i < A.size(); i++)
        ans[i] = c * A[i];
    return ans;
}
template<typename K>
vector<K> operator * (const vector<K>& A, const K& c) {
    vector<K> ans(A.size());
#pragma omp for
    for (int i = 0; i < A.size(); i++)
        ans[i] = A[i] * c;
    return ans;
}
template<typename K>
vector<K> operator * (const vector<K>& A, const vector<K>& B) {
    assert(A.size() == B.size());
    vector<K> C(A.size());
#pragma omp for
    for (int i = 0; i < A.size(); i++)
        C[i] = A[i] * B[i];
    return C;
}
template<typename K>
vector<K> operator / (const vector<K>& A, const K& c) {
    assert(c != 0);
    vector<K> ans(A.size());
#pragma omp for
    for (int i = 0; i < A.size(); i++)
        ans[i] = A[i] / c;
    return ans;
}
template<typename K>
vector<K> operator / (const vector<K>& A, const vector<K>& B) {
    assert(A.size() == B.size());
    vector<K> ans(A.size());
#pragma omp for
    for (int i = 0; i < A.size(); i++) 
    {
        assert(B[i] != 0);
        ans[i] = A[i] / B[i];
    }
    return ans;
}
template<typename K>
K norm2(const vector<K>& u) {
    K ans = 0;
    for (int i = 0; i < u.size(); i++)
        ans += u[i] * u[i];
    return sqrt(ans);
}
template<typename K>
vector<K> poject(const vector<K>& u, const vector<K>& v) {
    vector<K> proj(u.size());
    K c = 0;
    c += v dot u;
    K norm = u dot u;
    c /= (norm != 0) ? norm : 1;
    proj = c * u;
    return proj;
}
template<typename K>
vector<K> extending(const vector<K>& u, const vector<K>& v) {
    vector<K> tmp(u.size() + v.size());
    memcpy(&tmp[0], &u[0], sizeof(K) * u.size());
    memcpy(&tmp[u.size()], &v[0], sizeof(K) * v.size());
    return tmp;
}
template<typename K>
vector<vector<K>> vectorTranspose(const vector<K>& A)
{
    vector<vector<K>> B(A.size(), vector<K>(1, 0));
    for (int i = 0; i < A.size(); i++)
        B[i][0] = A[i];
    return B;
}
template<typename K>
vector<vector<K>> operator ^ (const vector<K>& A, matrixT)
{
    return vectorTranspose(A);
}

//---------2d vector operator----------

template<typename K>
vector<K> magicmatmul(const vector<K>& A, const vector<K>& B, int m, int n) {
    assert(A.size() == m);
    assert(B.size() == n);
    vector<K> C(m * n, 0);
#pragma omp for
    for (int i = 0; i < m; i++) 
        for (int j = 0; j < n; j++) 
            C[i * n + j] = A[i] * B[j];
    return C;
}
template<typename K>
vector<K> matmul(const vector<K>& A, const vector<K>& B, int m, int n, int k) {
    vector<K> C(m * k);
#pragma omp for
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            K val = 0;
            for (int t = 0; t < n; t++) {
#pragma omp atomic
                val += A[i * n + t] * B[t * k + j];
            }
            C[i * k + j] = val;
        }
    }
    return C;
}
template<typename K>
vector<vector<K>> operator + (const vector<vector<K>>& A, const vector<vector<K>>& B) {
    if (A.size() != B.size() && A[0].size() != B[0].size()) {


        throw length_error("[vector2d] + : the size is not matching!");
    }
    vector<vector<K>> C(A.size(), vector<K>(A[0].size(), 0));
#pragma omp for
    for (int i = 0; i < A.size(); i++)
        for (int j = 0; j < A[0].size(); j++)
            C[i][j] = A[i][j] + B[i][j];
    return C;
}
template<typename K>
vector<vector<K>> operator - (const vector<vector<K>>& A, const vector<vector<K>>& B) {
    if (A.size() != B.size() && A[0].size() != B[0].size())
    {


        throw length_error("[vector2d] - : the size is not matching!");
    }
    vector<vector<K>> C(A.size(), vector<K>(A[0].size(), 0));
#pragma omp for
    for (int i = 0; i < A.size(); i++)
        for (int j = 0; j < A[0].size(); j++)
            C[i][j] = A[i][j] - B[i][j];
    return C;
}
template<typename K>
vector<vector<K>> operator * (const vector<vector<K>>& A, const vector<vector<K>>& B) {
    if (A[0].size() != B.size())
    {


        throw length_error("[vector2d] * : the size is not matching!");
    }
    vector<vector<K>> C(A.size(), vector<K>(B[0].size(), 0));
#pragma omp for
    for (int i = 0; i < A.size(); i++)
        for (int j = 0; j < B[0].size(); j++) {
            K val = 0;
            for (int k = 0; k < A[0].size(); k++) {
#pragma omp atomic
                val += A[i][k] * B[k][j];
            }
            C[i][j] = val;
        }
    return C;
}
template<typename K>
vector<vector<K>> operator * (const vector<vector<K>>& A, const vector<K>& B) {
    if (A[0].size() != B.size()) {
        throw length_error("[vector2d] * : the size is not matching!");
    }
    vector<vector<K>> C(A.size(), vector<K>(1, 0));
#pragma omp for
    for (int i = 0; i < A.size(); i++) {
        K val = 0;
        for (int k = 0; k < B.size(); k++) {
#pragma omp atomic
            val += A[i][k] * B[k];
        }
        C[i][0] = val;
    }
    return C;
}
template<typename K>
vector<vector<K>> operator * (const vector<K>& B, const vector<vector<K>>& A) {
    if (A.size() != B.size()) {
        throw length_error("[vector2d] * : the size is not matching!");
    }
    vector<vector<K>> C(1, vector<K>(A[0].size(), 0));
    K val = 0;
#pragma omp for
    for (int i = 0; i < B.size(); i++) {
        val = 0;
        for (int k = 0; k < B.size(); k++) {
#pragma omp atomic
            val += B[k] * A[k][i];
        }
        C[0][i] = val;
    }
    return C;
}
template<typename K>
vector<vector<K>> operator * (const vector<vector<K>>& A, const K& c) {
    vector<vector<K>> C(A.size(), vector<K>(A[0].size(), 0));
#pragma omp parallel for
    for (int i = 0; i < A.size(); i++)
        for (int j = 0; j < A[0].size(); j++)
            C[i][j] = A[i][j] * c;
    return C;
}
template<typename K>
vector<vector<K>> operator * (const K& c, const vector<vector<K>>& A) {
    vector<vector<K>> C(A.size(), vector<K>(A[0].size(), 0));
#pragma omp parallel for
    for (int i = 0; i < A.size(); i++)
        for (int j = 0; j < A[0].size(); j++)
            C[i][j] = A[i][j] * c;
    return C;
}
template<typename K>
bool operator > (const vector<K>& A, const K& c) {
    if (find_if_not(A.begin(), A.end(), [c](K x) {return x > c; }) != A.end())
        return false;
    return true;
}
template<typename K>
bool operator < (const vector<K>& A, const K& c) {
    if (find_if_not(A.begin(), A.end(), [c](K x) {return x < c; }) != A.end())
        return false;
    return true;
}
template<typename K>
bool operator == (const vector<vector<K>>& A, const K& c) {
#pragma omp for
    for (int i = 0; i < A.size(); i++)
        if (find_if_not(A[i].begin(), A[i].end(), [c](K x) {return x == c; }) != A[i].end())
            return false;
    return true;
}
template<typename K>
vector<vector<K>> vectorTranspose(const vector<vector<K>>& A) {
    vector<vector<K>> B(A[0].size(), vector<K>(A.size(), 0));
#pragma omp for
    for (int i = 0; i < A.size(); i++)
        for (int j = 0; j < A[0].size(); j++)
            B[j][i] = A[i][j];
    return B;
}
template<typename K>
vector<vector<K>> operator ^ (const vector<vector<K>>& A, matrixT)
{
    return vectorTranspose(A);
}
template<typename K>
vector<vector<K>> operator ^ (const vector<vector<K>>& A, matrixP)
{
    return MATRIX_PESUDOINVERSE(A);
}
//outer product
template<typename K>
vector<vector<K>> operator cross (const vector<vector<K>>& A, const vector<vector<K>>& B) {
    if (A[0].size() != B.size()) {
        show_size(A);
        throw length_error("[vector outer product(cross)] the size of matrix is not match");
    }
    return (A ^ T) * B;
}
//-------2d vector algorithm 2----------

template<typename K>
K norm1(const vector<vector<K>>& A) {
    vector<K> L(A[0].size(), 0);
    K sum = 0;
    for (int j = 0; j < A[0].size(); j++) {
        sum = 0;
        for (int i = 0; i < A.size(); i++)
            sum += abs(A[i][j]);
        L[j] = sum;
    }
    return *max_element(L.begin(), L.end());
}
template<typename K>
K Least_sqaures(const vector<vector<K>>& A) {
    return (A * vectorTranspose(A))[0][0];
}
template<typename K>
K iter_error(const vector<K>& A, const vector<K>& B, unsigned int r) {
    K xa = accumulate(A->begin(), A->end(), K(0));
    K xb = accumulate(B->begin(), B->end(), K(0));
    return xa / pow(xb, r);
}
template<typename K>
K iter_error(const vector<vector<K>>& A, const vector<vector<K>>& B, unsigned int r) {
    K xa = 0;
    K xb = 0;
    for (int i = 0; i < A.size(); i++) {
        xa += accumulate(A[i].begin(), A[i].end(), K(0));
        xb += accumulate(B[i].begin(), B[i].end(), K(0));
    }
    return abs(xa) / pow(abs(xb), r);
}
template<typename K>
vector<vector<K>> identity_matrix(int m) {
    vector<vector<K>> A(m, vector<K>(m, 0));
    for (int i = 0; i < m; i++)
        A[i][i] = 1;
    return A;
}
template<typename K>
vector<int> COUNTPIVOT(const vector<vector<K>>& A, int sz) {
    vector<int> pivot_pos;
    int k = 0;
    for (int i = 0; i < A.size(); i++)
        for (int j = 0; j < sz; j++)
            if (A[i][j] != 0) {
                pivot_pos.push_back(j);
                break;
            }
    return pivot_pos;
}
template<typename K>
bool MATRIX_INF_NAN_CHECK(const vector<vector<K>>& A) {
    for (int i = 0; i < A.size(); i++)
    {
        for (int j = 0; j < A[i].size(); j++)
        {
            if (isnan(A[i][j])) return true;
            else if (isinf(A[i][j])) return true;
        }
    }
    return false;
}
template<typename K>
bool isTri(const vector<vector<K>>& A, float decimal) {
    if (A.size() != A[0].size())
    {

        throw invalid_argument("[isTri] : it is not a sqaure matrix");
    }
    for (int i = 0; i < A.size(); i++)
        for (int j = 0; j < i; j++)
            if (round(A[i][j] * decimal) / decimal != 0) return false;
            else A[i][j] = 0; //to avoid real number problem.
    return true;
}
template<typename K>
vector<K> vec2d2vec1d(const vector<vector<K>>& A) {
    if (A[0].size() == 1)
        return (A ^ T)[0];
    else if (A.size() == 1)
        return A[0];
    vector<K> B(A.size() * A[0].size());
    for (int i = 0; i < A.size(); i++)
        for (int j = 0; j < A[0].size(); j++)
            B[A[0].size() * i + j] = A[i][j];
    return B;
}
template<typename K>
vector<vector<K>> MATRIX_SUB(const vector<vector<K>>& A, int sr, int sc, int srsize, int scsize) {
    vector<vector<K>> B(srsize, vector<K>(scsize, 0));
    for (int i = sr; i < sr + srsize; i++)
        for (int j = sc; j < sc + scsize; j++)
            if (j < A[0].size() && i < A.size())
                B[i - sr][j - sc] = A[i][j];
            else {

                printf("SUB [%d,%d]->[%d,%d]", sr, sc, sr + srsize, sc + scsize);
                throw out_of_range("[MATRIX_SUB] : The sub size is out of range");
            }
    return B;
}
template<typename K>
vector<vector<K>> ADDROW(const vector<vector<K>>& mat, const vector<K>& var, int i) {
    if (var.size() != mat[0].size())
    {
        show_size(mat);
        show_size(var);
        throw length_error("[ADDROW] row size is not matching");
    }
    vector<vector<K>> A = mat;
    A.insert(A.begin() + i, var);
    return mat;
}
template<typename K>
vector<vector<K>> ADDCOL(const vector<vector<K>>& mat, const vector<K>& var, int i) {
    if (var.size() != mat.size())
    {
        show_size(mat);
        show_size(var);
        throw out_of_range("[ADDCOL] col size is not matching");
    }
    vector<vector<K>> A = mat;
#pragma omp for
    for (int j = 0; j < mat.size(); j++)
        A[j].insert(A[j].begin() + i, var[j]);
    return A;
}
template<typename K>
vector<vector<K>> MATRIXADDCOL(const vector<vector<K>>& A, const vector<vector<K>>& B) {
    if (A.size() != B.size())
    {
        throw out_of_range("[MATRIXADDCOL] col size is not matching");
    }
    vector<vector<K>> C(A.size(), vector<K>(A[0].size() + B[0].size(), 0));
#pragma omp for
    for (int i = 0; i < C.size(); i++) {
        memcpy(&C[i][0], &A[i][0], sizeof(K) * A[0].size());
        memcpy(&C[i][A[0].size()], &B[i][0], sizeof(K) * B[0].size());
    }
    return C;
}
template<typename K>
vector<vector<K>> ROWSWAP(const vector<vector<K>>& mat, int i, int j) {
    swap(mat[i], mat[j]);
}
template<typename K>
vector<vector<K>> ROWDELETE(const vector<vector<K>>& mat, int i) {
    vector<vector<K>> ANS = mat;
    ANS.erase(ANS.begin() + i);
    return ANS;
}
template<typename K>
vector<vector<K>> COLDELETE(const vector<vector<K>>& mat, int i) {
    vector<vector<K>> ANS = mat;
#pragma omp for
    for (int j = 0; j < ANS.size(); j++)
    {
        ANS[j].erase(ANS[j].begin() + i);
    }
    return ANS;
}
template<typename K>
bool diagonal_positive_check(const vector<vector<K>>& A)
{
    for (int i = 0; i < min(A.size(), A[0].siz()); i++)
        if (A[i][i] < 0) return false;
    return true;
}
template<typename K>
vector<vector<K>> Grim_Schmidt(const vector<vector<K>>& A) {
    vector<vector<K>> B = A ^ T;
    vector<vector<K>> U(B.size());
    U[0] = B[0];
    if (norm2(B[0]) != 0)
        U[0] = U[0] / norm2(B[0]);
    for (int i = 1; i < B.size(); i++) {
        U[i] = B[i];
        for (int j = 0; j < i; j++)
            U[i] = U[i] - poject(U[j], B[i]);
        if (norm2(U[i]) != 0)
            U[i] = U[i] / norm2(U[i]);
    }
    return U ^ T;
}
template<typename K>
vector<vector<K>> Power_method(const vector<vector<K>>& A)
{
    if (A.size() != A[0].size()) {

        throw invalid_argument("[Power_method] : A is not a square matrix");
    }
    vector<K> x(A.size());
    vector<K> p;
    K n = 0;
    transform(x.begin(), x.end(), x.begin(), [](K v) {return 1; });
    int k = 0;
    while (k++ < 1000)
    {
        p = A * x;
        n = *max_element(p.begin(), p.end());
        transform(p.begin(), p.end(), p.begin(), [n](K v) {return (n != 0) ? v / n : 0; });
        if (p == x) break;
        else if (iter_error(p, x, k) == 0) break;
        x = p;
    }
    printf("ル = %f\n", n);
    printf("\tv\n"); show_vector(x); printf("\n");
}
template<typename K>
vector<K> eigen_value(const vector<vector<K>>& A, bool show = false) {
    vector<K> eigenvalue(A.size());
    unsigned int k = 1;
    vector<vector<K>> A_ = A;
    vector<vector<K>> Q;
    vector<vector<K>> NQ;
    while (true)
    {
        Q = Grim_Schmidt(A);
        if (isnan(Q[0][0])) throw logic_error("[eigen_value] : nan value is detected!");
        else if (Q == NQ) break;
        else if (iter_error(NQ, Q, k++) == 0.0f) break;
        NQ = Q;
        A_ = (Q ^ T) * A_ * Q;
    }
    for (int i = 0; i < A_.size(); i++)
        eigenvalue[i] = A_[i][i];
    if (show) {
        printf("\tル\n");
        show_vector(eigenvalue); printf("\n");
    }
    return eigenvalue;
}
//upper triangular reduce form
template<typename K> //fix swap algorithm
vector<vector<K>> REDUCE(const vector<vector<K>>& A, bool show = false) {
    int k = 0;
    if (show) {
        printf("\tORIGIN\n");
        show_vector(A);
    }
    vector<vector<K>> B = A;
    while (k < min(B.size(), B[0].size())) {
        if (B[k][k] == 0) //if diagonal is a 0
        {
            int j = k;
            for (; j < B[0].size(); j++)
            {
                for (int i = k + 1; i < B.size(); i++)
                {
                    if (B[i][j] != 0)
                    {    //find not 0 rows in lower 
                        swap(B[k], B[i]); //swap row
                        break;
                    }
                }
                if (B[k][j] != 0) //if the diagonal is not 0
                    break;
            }
        }
#pragma omp for
        for (int i = k + 1; i < B.size(); i++)
        {
            if (B[i][k] != 0)
            {
                float r = log(abs(B[i][k])) - log(abs(B[k][k]));
                const char sign1 = SIGN(B[i][k]);
                const char sign2 = SIGN(B[k][k]);
                for (int j = 0; j < B[0].size(); j++)
                {
                    const char sign3 = SIGN(B[k][j]);
                    if (sign1 * sign2 * sign3 > 0)
                        B[i][j] = B[i][j] - exp(log(abs(B[k][j])) + r);
                    else
                        B[i][j] = B[i][j] + exp(log(abs(B[k][j])) + r);
                    if (abs(B[i][j]) < 0.00000001) //it should be the error problem
                        B[i][j] = 0;
                }
                B[i][k] = 0;//too avoid realnumber problem
            }
        }
        k++;
    }
    if (show) {
        printf("\tREDUCE FORM\n");
        show_vector(B);
    }
    return B;
}
//guassian elimination form
template<typename K>
vector<vector<K>> GRF(const vector<vector<K>>& A, bool show = false) {
    if (show)
        printf("\tOriginal\n"), show_vector(A), printf("\n");
    int k = 0;
    vector<vector<K>> B = A;
    while (k < min(B.size(), B[0].size())) {
        if (B[k][k] == 0) //if diagonal is a 0
        {
            int j = k;
            for (; j < B[0].size(); j++)
            {
                for (int i = k + 1; i < A.size(); i++)
                {
                    if (B[i][j] != 0)
                    {    //find not 0 rows in lower 
                        swap(B[k], B[i]); //swap row
                        break;
                    }
                }
                if (B[k][j] != 0) //if the diagonal is not 0
                    break;
            }
        }
#pragma omp for
        if (B[k][k] != 1 && B[k][k] != 0)
        {
            float r = log(abs(B[k][k]));
            const char sign1 = SIGN(B[k][k]);
            B[k][k] = 1; //to avoid real number problem
            for (int i = k + 1; i < B[0].size(); i++)
            {
                if (B[k][i] != 0)
                {
                    const char sign2 = SIGN(B[k][i]);
                    B[k][i] = sign1 * sign2 * exp(log(abs(B[k][i])) - r);
                }
            }

        }
        for (int i = k + 1; i < B.size(); i++)
        {
            if (B[i][k] != 0)
            {
                float r = log(abs(B[i][k])) - log(abs(B[k][k]));
                const char sign1 = SIGN(B[i][k]);
                const char sign2 = SIGN(B[k][k]);
                for (int j = 0; j < B[0].size(); j++)
                {
                    const char sign3 = SIGN(B[k][j]);
                    if (sign1 * sign2 * sign3 > 0)
                        B[i][j] = B[i][j] - exp(log(abs(B[k][j])) + r);
                    else
                        B[i][j] = B[i][j] + exp(log(abs(B[k][j])) + r);
                    if (abs(B[i][j]) < 0.00000001) //it should be the error problem
                        B[i][j] = 0;
                }
                B[i][k] = 0;//too avoid realnumber problem
            }
        }
        k++;
    }
    if (show) {
        printf("\tGAUSSIAN ELIMINAION\n");
        show_vector(A);
    }
    return B;
}
//row echelon reduce form
//sz is the distinguish cols index number for argument matrix
template<typename K>
vector<vector<K>> RREF(const vector<vector<K>>& A, int sz, bool show = false) {
    vector<vector<K>> R = GRF(A, show);
#pragma omp for
    for (int i = R.size() - 1; i >= 0; i--) //from the bottom
        for (int j = 0; j < sz; j++)//find the pivot untill sz 
            if (R[i][j] != 0) {  //if find pivot
                for (int k = i - 1; k >= 0; k--) { //find other rows
                    if (R[k][j] != 0) { //if the same col is not a 0
                        float p = R[k][j]; //multiplier
                        R[k][j] = 0; //avoid real number problem
                        for (int t = j + 1; t < R[0].size(); t++)
                            R[k][t] -= p * R[i][t];
                    }
                }
                break;
            }
    if (show)
    {
        printf("\tRREF\n");
        show_vector(R);
    }
    return R;
}
//[0] is P
//[1] is L
//[2] is U
template<typename K>
vector<vector<vector<K>>> PLU_decomposition(const vector<vector<K>>& A, bool show = false)
{
    int k = 0;
    vector<vector<K>> U = A;
    vector<vector<K>> L(U.size(), vector<K>(U.size(), 0));
    vector<vector<K>> P = identity_matrix<K>(U.size());
    float r = 0;
    const char sign1 = 0;
    const char sign2 = 0;
    const char sign3 = 0;
    if (show) {
        printf("\tORIGIN\n");
        show_vector(U);
    }
    while (k < min(U.size(), U[0].size())) {
        if (U[k][k] == 0) //if diagonal is a 0
        {
            for (int j = k; j < U[0].size(); j++)
            {
                for (int i = k + 1; i < U.size(); i++)
                {
                    if (U[i][j] != 0)
                    {    //find not 0 rows in lower 
                        swap(U[k], U[i]); //swap row
                        swap(P[k], P[i]);
                        swap(L[k], L[i]);
                        break;
                    }
                }
                if (U[k][j] != 0) //if the diagonal is not 0
                    break;
            }
        }
#pragma omp for
        for (int i = k + 1; i < U.size(); i++)
        {
            if (U[i][k] != 0)
            {
                r = log(abs(U[i][k])) - log(abs(U[k][k]));
                sign1 = SIGN(U[i][k]);
                sign2 = SIGN(U[k][k]);
                if (k < L[0].size()) //update lower triangular matrix
                    L[i][k] = exp(r) * sign1 * sign2;
                for (int j = 0; j < U[0].size(); j++)
                {
                    sign3 = SIGN(U[k][j]);
                    if (sign1 * sign2 * sign3 > 0)
                        U[i][j] = U[i][j] - exp(log(abs(U[k][j])) + r);
                    else
                        U[i][j] = U[i][j] + exp(log(abs(U[k][j])) + r);
                    if (abs(U[i][j]) < 0.00000001) //it should be the error problem
                        U[i][j] = 0;
                }
                U[i][k] = 0;//too avoid realnumber problem
            }
        }
        k++;
    }
    for (int i = 0; i < L.size(); i++)
        L[i][i] = 1;
    if (show) {
        printf("\n PLU decomposition form\n");
        printf("\tP\n"); show_vector(P);
        printf("\tL\n"); show_vector(L);
        printf("\tU\n"); show_vector(U);
        printf("\tA\n"); show_vector(P * L * U);
    }
    return { P,L,U };
}
template<typename K>
vector<K> JACOBI_SOLVE_METHOD(const vector<vector<K>>& A, const vector<K>& b, bool show = false)
{
    //if diagonally dominant matrix
    vector<vector<K>> D(A.size(), vector<K>(A[0].size(), 0));
    vector<vector<K>> invD(A.size(), vector<K>(A[0].size(), 0));    //inverse strict Diagonal triangular matrix
    vector<vector<K>> SLU(A.size(), vector<K>(A[0].size(), 0));      //negative strict Upper triangular matrix
    vector<vector<K>> B = b ^ T;
#pragma omp for
    for (int i = 0; i < A.size(); i++)
    {
        for (int j = 0; j < A.size(); j++)
        {
            if (i == j)
            {
                if (A[i][j] == 0) throw logic_error("[JACOBI_SOLVE_METHOD] : Diagonal is zero!");
                else invD[i][j] = 1 / A[i][j], D[i][j] = A[i][j];
            }
            else if (i < j)
            {
                SLU[i][j] = -A[i][j];
            }
            else
            {
                SLU[i][j] = -A[i][j];
            }
        }
    }
    vector<vector<K>> x(1, vector<K>(b.size(), 1));
    vector<vector<K>> xk(1, vector<K>(b.size(), 1));
    x = x ^ T;
    unsigned int r = 1;
    while (true)
    {
        xk = invD * (SLU * x + B);
        if (x == xk) break;
        else if (norm2(SHRINK(A * xk - B)) < 0.00001f) break;
        x = xk;
    }
    return SHRINK(x);
}
template<typename K>
vector<K> REDUCE_SOLVE_METHOD(const vector<vector<K>>& A, const vector<K>& b, bool show = false)
{
    if (A.size() != b.size())
        throw length_error("[REDUCE_SOLVE_METHOD] A rows and B rows size are not the same!");
    vector<vector<K>> Ab = ADDCOL(A, b, A[0].size());
    vector<vector<K>> ans = RREF(Ab, Ab[0].size() - 1, show);
    //find equation variables
    vector<K> temp;
    vector<K> x(A[0].size(), 0);
    for (int i = 0; i < ans[0].size() - 1; i++)
    {
        temp.clear();
        for (int j = i; j < ans[0].size() - 1; j++)
            if (ans[i][j] != 0)
                temp.push_back(ans[i][j]);
        if (temp.empty() && ans[i].back() != 0)
        {
            printf("[MATRIX_SOLUTION] : no solution\n");
            perror("MATRIX_SOLUTION()");
            return x;
        }
        else if (temp.empty() && ans[i].back() == 0)
            x[i] = 1;
    }
    //solve equation by fixed varibles
    for (int i = 0; i < ans[0].size() - 1; i++)
    {
        if (x[i] == 0) {
            x[i] = ans[i].back();
            for (int j = i + 1; j < ans[0].size() - 1; j++)
                if (x[j] != 0) x[i] -= ans[i][j] * x[j];
        }
    }
    return x;
}
template<typename K>
vector<K> MATRIX_SOLUTION(const vector<vector<K>>& mat, const vector<K>& var, bool show = false) {
    vector<K> x(mat.size(), 0);
    if (var.size() == mat.size())
    {
        if (MATRIX_INF_NAN_CHECK(mat))
        {
            printf("[MATRIX_SOLUTION] : no solution\n");
            perror("MATRIX_SOLUTION()");
            return x;
        }
        if (mat[0].size() == var.size())
        {
            K max = mat[0][0];
            for (int i = 0; i < mat.size(); i++)
            {
                max = abs(mat[i][i]);
                if (max == 0)
                    return REDUCE_SOLVE_METHOD(mat, var, show);
                for (int j = 0; j < mat[0].size(); j++) //check it is diagonally dominant matrix
                    if (i != j && max < abs(mat[i][j]))
                        return REDUCE_SOLVE_METHOD(mat, var, show);
            }
            return JACOBI_SOLVE_METHOD(mat, var, show);
        }
        else if (mat[0].size() >= var.size())
        {
            return REDUCE_SOLVE_METHOD(mat, var, show);
        }
    }
    return x;
}
template<typename K>
vector<vector<K>> MATRIX_INVERSE(const vector<vector<K>>& A, bool show = false) {
    if (A.size() != A[0].size())
    {

        throw out_of_range("[MATRIX_INVERE] : A size is not square");
    }
    vector<vector<K>> I = identity_matrix<K>(A.size());
    vector<vector<K>> B = MATRIXADDCOL(A, I);
    vector<vector<K>> INVA = RREF(B, A.size(), show);
    if (show) {
        printf("\tECHELON\n");
        show_vector(INVA);
    }
    INVA = MATRIX_SUB(INVA, 0, INVA[0].size() / 2, INVA[0].size() / 2, INVA[0].size() / 2);
    if (show) {
        printf("\tINVERSE\n");
        show_vector(INVA);
    }
    return INVA;
}
template<typename K>
vector<vector<K>> MATRIX_PESUDOINVERSE(const vector<vector<K>>& A, bool show = false) {
    return MATRIX_INVERSE(vectorTranspose(A) * A, show) * vectorTranspose(A);
}
template<typename K>
vector<vector<vector<K>>> EIGEN_DECOMPOSITION(const vector<vector<K>>& A, bool show = false) {
    if (A[0].size() != A.size())
    {

        throw out_of_range("[DECOMPOSITION] : it is not a square matrix!");
    }
    vector<vector<K>> ニ;
    vector<vector<K>> V;
    vector<K> zero(A.size(), 0);
    vector<K> ル = eigen_value(A, show);
    bool zero_v = true;
    if (show)
    {
        printf("\tA\n"), show_vector(A);
    }
    sort(ル.begin(), ル.end(), greater<K>());
    for (int i = 0; i < ル.size(); i++)
    {
        ニ = identity_matrix<K>(A.size());
        ニ = ニ * ル[i]; //to avoid real number problem
        vector<K> v = MATRIX_SOLUTION(A - ニ, zero);
        for (int i = 0; i < v.size(); i++)
            if (v[i] != 0) {
                zero_v = false;
                break;
            }
        V.push_back(v);
    }
    ニ = identity_matrix<K>(ニ.size());
    for (int i = 0; i < A.size(); i++)
        ニ[i][i] = ル[i];
    V = vectorTranspose(V);

    if (show) {
        printf("\tニ\n"), show_vector(ニ);
        printf("\tV\n"), show_vector(V);
    }
    vector< vector< vector<K> > > Vニ;
    Vニ.push_back(V);
    Vニ.push_back(ニ);
    return Vニ;
}

//--------2d vector user operator---------

template<typename K>
vector<vector<K>> operator ^ (const vector<vector<K>>& A, int pow)
{
    vector<vector<K>> B = A;
    if (pow == -1) return MATRIX_INVERSE(A);
    else if (pow < 0) {
        printf("A^%d\n", pow);
        throw overflow_error("[vector2d operator] ^ : unknown exponent is input!");
    }
    for (int i = 0; i < pow; i++)
        B = B * B;
    return B;
}
template<typename K>
vector<vector<K>>& operator CLUMPLE (vector<vector<K>>& A, const vector<K>& B)
{
    if (A.size() * A[0].size() != B.size()) {


        throw length_error("[vector2d] CLUMPLE : total A size and B size are not the same!");
    }
    for (int i = 0; i < A.size(); i++)
        for (int j = 0; j < A[0].size(); j++)
            A[i][j] = B[A[0].size() * i + j];
    return A;
}

#endif