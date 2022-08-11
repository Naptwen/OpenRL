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
	A.learning_set(0, 0.001, 0, KL_DIVERGENCE, 100, NADAM);
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
