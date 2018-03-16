#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
//N is the no. of nodes in hidden layer
#define N 8
#define epsilon 0.01
#define epochs 1500


//Hidden units=N
/************************************************************************************************/
//Global
int X,Y;		//X stores the last example index in train set and Y in test set
float lrate=0.001;	//Learning rate
int raw_train_data[2500][17];
int raw_test_data[2500][17];
int train_label[2500][1];
int test_label[2500][1];
int new_trainLabel[2500][10];//Converted to 0 and 1 form
int new_testLabel[2500][10];
float outError[1][10];
float theta1[17][8];		//included bias
float theta2[9][10];
float layer1[1][9];
float before_layer1[1][9];
float netj[1][9];
float layer2[1][10];
float before_layer2[1][10];
float netk[1][10];
float w2[10][9];
float w1[8][17];
float deltaj[1][8];
float fdash[1][10];		//f' of output layer
float delta[1][10];	//delta
int output[1][10];
int symbol1[2500][10];		//For output symbols 0,1
int symbol2[2500][10];		//For label symbols 0,1
///////////////////////////////////////////////////////////////////////////////////////////////////


/***********************************************************************************************/
//File reading

void train_read(){
	FILE *f=fopen("train1.txt","r");
	int i,j;
	for(i=0;i<2500;i++){
		for(j=0;j<17;j++){
			if(j==0){
				fscanf(f,"%d",&train_label[i][j]);
			}
			else{
				fscanf(f,"%d",&raw_train_data[i][j]);
			}
		}
	}
	fclose(f);
}
void test_read(){
	FILE *f=fopen("test.txt","r");
	int i,j;
	for(i=0;i<2500;i++){
		for(j=0;j<17;j++){
			if(j==0){
				fscanf(f,"%d",&test_label[i][j]);
			}
			else{
				fscanf(f,"%d",&raw_test_data[i][j]);
			}
		}
	}
	fclose(f);
}
void print_mat(char c){
	int i,j;
	if(c=='t'){
		for(i=0;i<2500;i++){
			for(j=0;j<17;j++){
				printf("%d ",raw_train_data[i][j]);
			}
			printf("\n");
		}
	}
	else if(c=='l'){
		for(i=0;i<2500;i++){
			printf("%d\n",train_label[i][0]);
		}
	}
	else if(c=='L'){
		for(i=0;i<2500;i++){
			printf("%d\n",test_label[i][0]);
		}
	}
	else{
		for(i=0;i<2500;i++){
		for(j=0;j<17;j++){
			printf("%d ",raw_test_data[i][j]);
		}
		printf("\n");
	}
	}
}
/********************************************************************************************************************/
//Add bias
void add_bias(){
	int i,j,flag;
	for(i=0;i<2500;i++){
		flag=0;
		for(j=0;j<17;j++){
			if(raw_train_data[i][j]!=0){
				flag=1;break;
			}
		}
		if(flag==0){
			X=i;

			break;
		}
		else{raw_train_data[i][0]=1;}
	}

	for(i=0;i<2500;i++){
		flag=0;
		for(j=0;j<17;j++){
			if(raw_test_data[i][j]!=0){
				flag=1;break;
			}
		}
		if(flag==0){
			Y=i;
			break;
		}
		else{raw_test_data[i][0]=1;}
	}
}
/*******************************************************************************************************************/
//sigmoid
float sigmoid(float x)
{
     float exp_value;
     float return_value;
     exp_value = exp((double) -x);
     return_value = 1 / (1 + exp_value);

     return return_value;
}
//derivative of sigmoid
float sigmoid_bar(float x){
	return sigmoid(x)*(1-sigmoid(x));
}
/*********************************************************************************************************************/
//Generate weights
void fill_weight(){
	srand(time(NULL));
	int i,j,x;
	for(i=0;i<17;i++){
		for(j=0;j<N;j++){
			 theta1[i][j]=(float)(rand()%3-1)/100.0;
			 //printf("%f ",theta1[i][j]);
		}
		//printf("\n");
	}
	//printf("------------------theta1---------------------\n");
	for(i=0;i<=N;i++){
		for(j=0;j<10;j++){
			 theta2[i][j]=(float)(rand()%5+1)/100.0;
			// printf("%f ",theta2[i][j]);
		}
		//printf("\n");
	}
	//printf("------------------theta2---------------------\n");
	/*for(i=0;i<17;i++){
		for(j=0;j<N;j++){
			printf("%d ",theta1[i][j]);
		}
		printf("\n");
	}*/

}
void convert_out(){
	int k,m=0,j;
	for(j=0;j<10;j++){output[0][j]=0;}
	float ma=netk[0][0];
	for(k=0;k<10;k++){
		if(netk[0][k]>ma){
			ma=netk[0][k];
			m=k;
		}
	}
	//printf("%d\n",m);
	for(j=0;j<10;j++){output[0][j]=0;}
	output[0][m]=1;
}
//Convert labels to matrices of 0 1
void convert_label(int a[][1],int r){
	int i,j;
	for(i=0;i<2500;i++){
		for(j=0;j<10;j++){
			new_trainLabel[i][j]=0;
			new_testLabel[i][j]=0;
		}
	}
	for(i=0;i<X;i++){
		if(r==0){
			/* ITS TRAIN LABEL*/
			new_trainLabel[i][a[i][0]-1]=1;
		}
		else{
			//ITS TEST LABEL
			new_testLabel[i][a[i][0]-1]=1;
		}

	}
}

void calculate_error(int i){
	int j;
	for(j=0;j<10;j++){
		outError[0][j]=new_trainLabel[i][j]-layer2[0][j];
		//printf("%d -%f=%f\n",new_trainLabel[i][j],layer2[0][j],outError[0][j]);
	}
}
void calculate_layer(){
	int i,j,k,e;
	float sum;
	convert_label(train_label,0);	//new_trainLabel
	//convert_label(test_label,1);	//new_testLabel
for(e=0;e<epochs;e++){
	for(i=0;i<X;i++){
//printf("\n");
		for(j=0;j<N;j++){
			sum=0;
			for(k=0;k<17;k++){
				sum+=(float)raw_train_data[i][k]*theta1[k][j];
				//printf("%f ",theta1[k][j]);
			}
			layer1[0][j+1]=sum;
			before_layer1[0][j+1]=sum;
			//printf("\n");
		}
		//printf("\n");
		/*for(j=0;j<=N;j++){
			printf("%f ",layer1[0][j]);
		}*/
//break;
		for(j=0;j<N;j++){
			layer1[0][j+1]=sigmoid(layer1[0][j+1]);	//Output of hidden layer
		}
		layer1[0][0]=1;
		before_layer1[0][0]=1;		//Add bias
		/*for(j=0;j<=N;j++){
			printf("%f ",layer1[0][j]);
		}*/


		int l;
		for(l=0;l<1;l++){
			for(j=0;j<10;j++){
				sum=0;
				for(k=0;k<=N;k++){
					sum+=layer1[l][k]*theta2[k][j];
				}
				layer2[l][j]=sum;
				before_layer2[l][j]=sum;
			}
		}
		/*for(j=0;j<=N;j++){
			printf("%f ",layer2[0][j]);
		}*/
		for(k=0;k<10;k++){
			layer2[0][k]=sigmoid(layer2[0][k]);	//Output of output layer
		}
		/*for(j=0;j<10;j++){
			printf("%f ",layer2[0][j]);
		}*/


		//Convert to 0 1
		//convert_out();
		/*for(j=0;j<10;j++){printf("%d",output[0][j]);}
			printf("\n");*/
		//output now contains 0 and 1
		//Error!!!!! Calculate
		calculate_error(i);
		//outError matrix now available

		for(k=0;k<10;k++){
			fdash[0][k]=sigmoid_bar(before_layer2[0][k]);
		}
		// f'(net) over
		//////////////////////////////////////////
		for(k=0;k<10;k++){
			delta[0][k]=outError[0][k]*fdash[0][k];
			//printf("%f \n",outError[0][k]);
		}

		//Multiply with layer1(1x6) delta(1x10)

		for(l=0;l<10;l++){
			for(j=0;j<=N;j++){
				sum=0;
				for(k=0;k<1;k++){
					//printf("%f ",sum);
					sum+=delta[k][l]*layer1[k][j]*lrate;
					//printf("%f ",delta[k][l]);
				}
				w2[l][j]=sum;
				//printf("%f ",w2[l][j]);
			}
			//printf("\n");
		}
		//printf("------------------------wj-------------------------\n");


		//w2 is 10x6 matrix
		for(j=0;j<=N;j++){
			for(k=0;k<10;k++){
				theta2[j][k]+=w2[k][j];
			}
		}
		/*for(j=0;j<=N;j++){
			for(k=0;k<10;k++){
				printf("%f ",theta2[j][k]);
			}
			printf("\n");
		}*/
		int r;
		for(j=1;j<N+1;j++){
		 sum=0;
			for(r=0;r<10;r++){
				sum+=delta[0][r]*theta2[j][r]*sigmoid_bar(before_layer1[0][j]);
			}
			//printf("\n");
			deltaj[0][j-1]=sum;

		}
		/*for(j=0;j<N;j++){
			printf("%f  ",deltaj[0][j]);
		}*/
		//printf("--------------------delta-------------------------\n");
		/*for(j=1;j<N+1;j++){
			printf("%f\n",sigmoid_bar(layer1[0][j]));
		}*/
		/*for(j=0;j<N;j++){
			printf("%f ",deltaj[0][j]);
		}*/
		//Multiply with xi input (1x17) deltaj is(1x5) 5x17
		for(j=0;j<N;j++){
			for(k=0;k<17;k++){
				w1[j][k]=deltaj[0][j]*raw_train_data[i][k]*lrate;
			}
		}


		//print w1




		//w1 calculated 5x17
		//w2 is 10x6 including bias,ignore bias
		//theta1 17x5 theta2 6x10
		//update weights
		for(j=0;j<N;j++){
			for(k=0;k<17;k++){
				theta1[k][j]+=w1[j][k];
				//printf("%f ",w1[j][k]);
			}
			//printf("\n");
		}



		//break;
	}

	//break;
}
/*for(j=0;j<N;j++){
	for(k=0;k<17;k++){
		printf("%f ",theta1[k][j]);
	}
	printf("\n");
}
printf("\n\n");
for(j=0;j<=N;j++){
	for(k=0;k<10;k++){
		printf("%f ",theta2[j][k]);
	}
	printf("\n");
}*/

}

void test_perceptron1(){
	//theta1 and theta2 are the weights
	//raw_test_data and test_label new_testLabel
	//netj holds 1st layer and netk holds second layer
	int i,j,k,count=0;
	float sum;
	//("%d\n",Y);
	for(i=0;i<Y;i++){
		//printf("\nVal=%d\n",test_label[i][0]);
		for(j=0;j<N;j++){
			sum=0;
			for(k=0;k<17;k++){
				sum+=(float)raw_test_data[i][k]*theta1[k][j];
			}
			netj[0][j+1]=sum;
			//printf("%f ",netj[0][j+1]);
		}
		//break;
		for(k=0;k<17;k++){
			//printf("%d ",raw_test_data[i][k]);
		}
		//printf("\n");
		for(j=0;j<N;j++){
			//printf("%f ",netj[0][j+1]);
			netj[0][j+1]=sigmoid(netj[0][j+1]);	//Output of hidden layer
			//printf("%f ////",netj[0][j+1]);
		}
		//printf("\n");
		netj[0][0]=1;



		int l;

		for(l=0;l<1;l++){
			for(j=0;j<10;j++){
				sum=0;
				for(k=0;k<=N;k++){
					sum+=netj[l][k]*theta2[k][j];
				}
				netk[l][j]=sum;
				//printf("%f ",sum);
			}
			//printf("\n");
		}


		for(k=0;k<10;k++){
			netk[0][k]=sigmoid(netk[0][k]);	//Output of output layer
			//printf("%f ",netk[0][k]);
		}

		convert_out();
		//output now available

		for(k=0;k<10;k++){
			if(output[0][k]==1 && k+1==test_label[i][0]){
				//printf("%d\n",k+1);
				count++;
			}

		}

//break;
	}
	printf("Accuracy= %f\n",(float)count*100.0/(float)Y);

}
/*******************************************************************************************************************/
/*MAIN*/
int main(){
	printf("Sum of Squared Error Loss Used \n");
	train_read();	//read training data
	test_read();	//read testing data
	add_bias();		//add bias unit
	fill_weight();
	//printf("%f ",sigmoid(81.00));
	calculate_layer();
	/*for(int i=0;i<17;i++){
		for(int j=0;j<8;j++){
			printf("%f ",theta1[i][j]);
		}
		printf("\n");
	}
	for(int i=0;i<9;i++){
		for(int j=0;j<10;j++){
			printf("%f ",theta2[i][j]);
		}
		printf("\n");
	}*/
	test_perceptron1();
	printf("No. of hidden units= %d\n",N);
	printf("No. of epochs =%d\n",epochs);
	printf("Learning Rate= %f\n",lrate);
	//printf("%d",X);printf("%d",Y);
	//print_mat('t');
	//print_mat('o');
	//print_mat('L');
	//printf("%d\n",sigmoid(-1));

	return 0;
}
