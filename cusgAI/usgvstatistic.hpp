#pragma once
#include "usgvmatrix.hpp"
#include <cmath>
// sum
template <typename K>
K sum(const vector<K> list)
{
    K sum = 0;
#pragma omp for
    for(int i = 0; i < list.size(); i++)
#pragma omp atomic
        sum += list[i];
    return sum;
}
// average
template <typename K>
K avg(const vector<K> list)
{
    K sum = 0;
#pragma omp for
    for (int i = 0; i < list.size(); i++)
#pragma omp atomic
        sum += list[i];
    return sum/list.size();
}
//std
template <typename K>
K std(const vector<K> list, const bool population= false)
{
    float m = avg(list);
    float sum = 0;
    for (int i = 0; i < list.size(); i++)
        sum += std::pow(list[i], 2);
    return (population) ? sum / list.size() : sum / (list.size() - 1);
}
