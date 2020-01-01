<h1>TakiMatrix</h1>

TakiMatrix is matrix processing framework providing maximized concurrency
autuor: Justin Kim

<h4>features</h4>
* fast execution of matrix operations using CUDA
* searches for instruction level concurrency and executes them simultaneously
* implements out-of-order execution and register renaming to maximize throughput

<h4>register renaming</h4>
removes WAR, WAW relations

```c++
	matrix = matrix_a + matrix_b; //1
	matrix_c = matrix*matrix_a; //2
	matrix = matrix_a*matrix_d; //3
```
operation 2 must come after operation 1

matrix object of operation 1 and operation 3 is seperated internally, so execution order of operation 1 and 3 can be shuffled

```c++
	matrix = matrix_a + matrix_b; //1
	matrix_c = matrix*matrix_a; //2
	matrix = matrix_a; //3
```

execution order of operation 2 and 3 can be swapped since
'matrix' in operation 2 and 'matrix' in operation 3 is seperated internally.

<h4>branches</h4>
equality operators from matrix and matrix::matrix_ptr() will wait for following matrix to be executed before returning
