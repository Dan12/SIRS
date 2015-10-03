//GSL Matrix Multiplication

#include <stdio.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

#define GSL_RANGE_CHECK_OFF

void printMatrix(gsl_matrix * m);
void matrixTests();
void mmAB(gsl_matrix * A, gsl_matrix * B, gsl_matrix * C);
void mmAtB(gsl_matrix * A, gsl_matrix * B, gsl_matrix * C);
void mmABt(gsl_matrix * A, gsl_matrix * B, gsl_matrix * C);
void mInv(gsl_matrix * A);

int main (void){

  double a[] = { 1, 2,
                 3, 4 };

  double b[] = { 1, 0, 1,
  	  	  	  	 0, 2, 0};

  gsl_matrix_view A = gsl_matrix_view_array(a, 2, 2);
  gsl_matrix_view B = gsl_matrix_view_array(b, 2, 3);
  gsl_matrix* C;
  C = gsl_matrix_alloc(2, 3);

  /* Compute C = A B */

  gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,
                  1.0, &A.matrix, &B.matrix,
                  0.0, C);

  printMatrix(C);

  //mInv(&A.matrix);

  matrixTests();

  gsl_matrix_free(C);

  return 0;
}

void matrixTests(){
	double a[] = {	1, 2,
					3, 4};
	double b[] = {	1, 2, 3,
					2, 1, 1};
	gsl_matrix_view A = gsl_matrix_view_array(a,2,2);
	gsl_matrix_view B = gsl_matrix_view_array(b,2,3);
	gsl_matrix * C = gsl_matrix_alloc(2,3);

	printMatrix(&A.matrix);
	printMatrix(&B.matrix);
	printMatrix(C);

	mmAB(&A.matrix,&B.matrix,C);
	printMatrix(C);

	mmAtB(&A.matrix,&B.matrix,C);
	printMatrix(C);

	mInv(&A.matrix);

	gsl_matrix_free(&A.matrix);
	gsl_matrix_free(&B.matrix);
	gsl_matrix_free(C);
}

void mmAB(gsl_matrix * A, gsl_matrix * B, gsl_matrix * C){
	gsl_matrix_set_zero(C);
	gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, A, B, 0.0, C);
}

void mmAtB(gsl_matrix * A, gsl_matrix * B, gsl_matrix * C){
	gsl_matrix_set_zero(C);
	gsl_blas_dgemm (CblasTrans, CblasNoTrans, 1.0, A, B, 0.0, C);
}

void mmABt(gsl_matrix * A, gsl_matrix * B, gsl_matrix * C){
	gsl_matrix_set_zero(C);
	gsl_blas_dgemm (CblasNoTrans, CblasTrans, 1.0, A, B, 0.0, C);
}

void mInv(gsl_matrix * A){
	int s;

	gsl_matrix * AA = gsl_matrix_alloc(A->size1, A->size2);
	gsl_matrix_memcpy(AA,A);

	gsl_permutation * p = gsl_permutation_alloc (A->tda);

	gsl_matrix * B = gsl_matrix_alloc(A->size1,A->size2);

	gsl_linalg_LU_decomp (AA, p, &s);
	printMatrix(AA);

	gsl_linalg_LU_invert (AA, p, B);

	gsl_matrix * C = gsl_matrix_alloc(A->size1, A->size2);
	mmAB(A,B,C);

	printMatrix(AA);
	printMatrix(A);
	printMatrix(B);
	printMatrix(C);
}

void printMatrix(gsl_matrix * m){
	int i,j;
	for(i = 0; i < m->size1; i++){
		for(j = 0; j < m->size2; j++){
			printf("%g ",gsl_matrix_get(m,i,j));
		}
		printf("\n");
	}
	printf("-------\n");
}
