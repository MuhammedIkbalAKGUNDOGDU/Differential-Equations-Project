#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>


#define LEARNING_RATE 0.01
#define MAX_EPOCHS 1000
#define DATASET_SIZE 200
#define DICTIONARY_SIZE 1000
#define TRAIN_SPLIT 0.8
#define INPUT_SIZE DICTIONARY_SIZE
#define OUTPUT_SIZE 1
#define BETA1 0.9
#define BETA2 0.999
#define EPSILON 1e-8

int main()
{
	char **veriKumesi1, **veriKumesi2,*dataset,*dataset2,**uniondataset,**dictionary,*temp;
	int i,j,dataset1Count,dataset2Count,hotVektorCount,**hotvektor1,**hotvektor2;

	veriKumesi1 = (char**) calloc(DICTIONARY_SIZE, sizeof(char*));
	veriKumesi2 = (char**) calloc(DICTIONARY_SIZE, sizeof(char*));
	uniondataset = (char**) calloc(DICTIONARY_SIZE, sizeof(char*));
	dictionary = (char**) calloc(DICTIONARY_SIZE, sizeof(char*));
	hotvektor1 = (char**) calloc(DICTIONARY_SIZE, sizeof(char*));
	hotvektor2 = (char**) calloc(DICTIONARY_SIZE, sizeof(char*));
	dataset = (char*) calloc(30, sizeof(char*));
	dataset2 = (char*) calloc(30, sizeof(char*));

	for(i=0; i<1000; i++) {
        hotvektor1[i] = (char*) calloc(DICTIONARY_SIZE, sizeof(char));
        hotvektor2[i] = (char*) calloc(DICTIONARY_SIZE, sizeof(char));
		veriKumesi1[i] = (char*) calloc(DICTIONARY_SIZE, sizeof(char));
		veriKumesi2[i] = (char*) calloc(DICTIONARY_SIZE, sizeof(char));
		uniondataset[i] = (char*) calloc(DICTIONARY_SIZE, sizeof(char));
		dictionary[i] = (char*) calloc(DICTIONARY_SIZE, sizeof(char*));
	}
    dataset = "dataset1.txt";
    dataset1Count = fileProgres(veriKumesi1,dataset);
    lowercase(veriKumesi1,dataset1Count);

    dataset2 = "dataset2.txt";
    dataset2Count = fileProgres(veriKumesi2,dataset2);
    lowercase(veriKumesi2,dataset2Count);

    hotVektorCount = unionDataset(veriKumesi1,veriKumesi2,uniondataset,dataset1Count,dataset2Count);
    dictionaryCreate(hotVektorCount,dictionary,uniondataset);

     int sentence1,sentence2,sentence;
     sentence1 = hotVektorCreater(hotVektorCount,dataset2,dictionary,hotvektor2);
     sentence2 = hotVektorCreater(hotVektorCount,dataset,dictionary,hotvektor1);

     sentence = sentence1+sentence2;


    //gradient descent
    double **dataset3 = (double**)calloc(hotVektorCount , sizeof(double*));
    for (i = 0; i < hotVektorCount; i++) {
        dataset3[i] = (double*)calloc((INPUT_SIZE + 1) , sizeof(double));
        for (j = 0; j < sentence; j++) {
            if(j<sentence/2 && hotvektor1[i][j]==1){
                dataset3[i][j] = 1;

            }else if(hotvektor2[i][j-sentence1]==1){
                dataset3[i][j] = 1;
            }
        }
        dataset3[i][sentence] = i < sentence / 2 ? 1.0 : -1.0; // Ýlk yarýsý sýnýf A, diðer yarýsý sýnýf B
    }
    for(i=0;i<hotVektorCount;i++){
        for(j=0;j<sentence;j++){
            printf("%.f ",dataset3[i][j]);
        }
        printf("\n");
    }

    // Eðitim ve test kümesine böl
    int training_size = (int)(TRAIN_SPLIT * sentence);
    double **training_set = dataset3;
    double **test_set = dataset3 + training_size;

    // Parametreleri baþlat
    double *weights = (double*)malloc((INPUT_SIZE + 1) * sizeof(double));
    initialize_weights(weights);

    // Eðitim
    printf("\n\n\nGD\n");
    train(training_set, training_size, weights);

    // Test
    int correct_predictions = 0;
    for (int i = 0; i < sentence - training_size; i++) {
        double input[INPUT_SIZE];
        for (int j = 0; j < INPUT_SIZE; j++) {
            input[j] = test_set[i][j];
        }

        double output;
        forward_propagation(input, weights, &output);

        int predicted_class = output >= 0 ? 1 : -1;
        int true_class = test_set[i][INPUT_SIZE] == 1.0 ? 1 : -1;

        if (predicted_class == true_class) {
            correct_predictions++;
        }
    }

    // Test baþarýsý
    double accuracy = (double)correct_predictions / (sentence - training_size);
    printf("Test Basarisi: %lf\n", accuracy);

    // SGD
    // Eðitim

    printf("\n\n\n SGD");
    initialize_weights(weights);
    train_sgd(training_set, training_size, weights);

    // Test
    correct_predictions = 0;
    for (int i = 0; i < sentence - training_size; i++) {
        double input[INPUT_SIZE];
        for (int j = 0; j < INPUT_SIZE; j++) {
            input[j] = test_set[i][j];
        }

        double output;
        forward_propagation(input, weights, &output);

        int predicted_class = output >= 0 ? 1 : -1;
        int true_class = test_set[i][INPUT_SIZE] == 1.0 ? 1 : -1;

        if (predicted_class == true_class) {
            correct_predictions++;
        }
    }

    // Test baþarýsý
    accuracy = (double)correct_predictions / (sentence - training_size);
    printf("Test Basarisi: %lf\n", accuracy);


    // ADAM
    // Eðitim
    printf("\n\n\n ADAM");
    initialize_weights(weights);
    train_adam(training_set, training_size, weights);

    // Test
     correct_predictions = 0;
    for (int i = 0; i < sentence - training_size; i++) {
        double input[INPUT_SIZE];
        for (int j = 0; j < INPUT_SIZE; j++) {
            input[j] = test_set[i][j];
        }

        double output;
        forward_propagation(input, weights, &output);

        int predicted_class = output >= 0 ? 1 : -1;
        int true_class = test_set[i][INPUT_SIZE] == 1.0 ? 1 : -1;

        if (predicted_class == true_class) {
            correct_predictions++;
        }
    }

    // Test baþarýsý
     accuracy = (double)correct_predictions / (sentence - training_size);
    printf("Test Basarisi: %lf\n", accuracy);


    return 0;
}
void train_adam(double **training_set, int training_size, double *weights) {
    double m[INPUT_SIZE + 1] = {0};  // Birinci moment
    double v[INPUT_SIZE + 1] = {0};  // Ýkinci moment
    int t = 0;

    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        double total_loss = 0.0;

        for (int i = 0; i < training_size; i++) {
            t++;

            double input[INPUT_SIZE];
            for (int j = 0; j < INPUT_SIZE; j++) {
                input[j] = training_set[i][j];
            }

            double output;
            forward_propagation(input, weights, &output);

            double target = training_set[i][INPUT_SIZE];
            total_loss += pow(output - target, 2);

            backward_propagation(input, weights, output, target);

            // Adam ile her örnek için aðýrlýklarý güncelle
            for (int j = 0; j <= INPUT_SIZE; j++) {
                m[j] = BETA1 * m[j] + (1 - BETA1) * weights[j];
                v[j] = BETA2 * v[j] + (1 - BETA2) * pow(weights[j], 2);

                double m_hat = m[j] / (1 - pow(BETA1, t));
                double v_hat = v[j] / (1 - pow(BETA2, t));

                weights[j] -= LEARNING_RATE * m_hat / (sqrt(v_hat) + EPSILON);
            }
        }

        if (epoch % 100 == 0) {
            printf("Epoch %d, Loss: %lf\n", epoch, total_loss);
        }
    }
}

void train_sgd(double **training_set, int training_size, double *weights) {
    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        double total_loss = 0.0;

        for (int i = 0; i < training_size; i++) {
            double input[INPUT_SIZE];
            for (int j = 0; j < INPUT_SIZE; j++) {
                input[j] = training_set[i][j];
            }

            double output;
            forward_propagation(input, weights, &output);

            double target = training_set[i][INPUT_SIZE];
            total_loss += pow(output - target, 2);

            backward_propagation(input, weights, output, target);

            // SGD ile her örnek için aðýrlýklarý güncelle
            for (int j = 0; j <= INPUT_SIZE; j++) {
                weights[j] -= LEARNING_RATE * weights[j];
            }
        }

        if (epoch % 100 == 0) {
            printf("Epoch %d, Loss: %lf\n", epoch, total_loss);
        }
    }
}

int unionDataset(char** verikumesi1,char** verikumesi2,char** uniondataset,int veri1,int veri2){
    int i ,j,k, control = 1,result,hotvektoCounter = 0,control2 = 1;

    for(i = 0; i<veri1;i++){
        for(k=i; k<veri1;k++){
            if(strcmp(verikumesi1[k],verikumesi1[i]) == 0 && i != k)
            {
               control = 0;

            }
        }
        if (control == 1){
           for(j=0; j<veri2;j++){
            if(strcmp(verikumesi1[i],verikumesi2[j])==0){
                result = 0;
            }
            }
            if(result == 0)
            {
                uniondataset[hotvektoCounter] = verikumesi1[i];
                hotvektoCounter++;
            }else{
                uniondataset[hotvektoCounter] = verikumesi1[i];
                hotvektoCounter++;
            }
        }
        control = 1;
        result = 1;
    }
    for(i = 0; i<veri2 ; i++){
    control2 =1;
    result = 1;
         for(k=i; k<veri2;k++){
            if(strcmp(verikumesi2[k],verikumesi1[i]) == 0 && i != k)
            {
               control2 = 0;
            }
        }
        if (control2 == 1){
           for(j=0; j<hotvektoCounter;j++){
                if(strcmp(verikumesi2[i],uniondataset[j])==0)
                {
                    result = 0;
                }
            }
            if(result == 0)
            {

            }else{
                uniondataset[hotvektoCounter] = verikumesi2[i];
                hotvektoCounter++;
            }
        }
    }
    return hotvektoCounter;
}

void dictionaryCreate(int dictionarySize,char** dictionary,char** uniondataset){
    char temp[20];
    int i,j;

    for(i=0;i<dictionarySize;i++){
        strcpy(dictionary[i],uniondataset[i]);
    }

     for (i = 0; i < dictionarySize - 1; i++) {
        for (j = 0; j < dictionarySize - 1; j++) {
            if (strcmp(dictionary[j], dictionary[j + 1]) > 0) {
                strcpy(temp, dictionary[j]);
                strcpy(dictionary[j], dictionary[j + 1]);
                strcpy(dictionary[j + 1], temp);
            }
        }
}
}

int fileProgres(char **verikumesi, char *file){
    FILE *dosya = fopen(file, "r");
    char kelime[50],karakter;
    int kelimeIndex = 0,i=0,j;
    if (dosya == NULL) {
        printf("File Open Error!\n");
        return 0;
    }

    while ((karakter = fgetc(dosya)) != EOF) {
        if (isalnum(karakter)) {
            kelime[kelimeIndex++] = karakter;
        } else if (kelimeIndex > 0) {

            strcpy(verikumesi[i],kelime);
            i++;
            for( j = 0 ; j< kelimeIndex ; j++)
            {
                kelime[j] = '\0';
            }
            kelimeIndex = 0;
        }
    }
    fclose(dosya);
    return i;
}

void lowercase(char** dictionary,int count){
    int i,j;
    for(i=0;i<count;i++)
    {   j=0;
        while(dictionary[i][j] != '\0')
        {
            dictionary[i][j] = tolower(dictionary[i][j]);
            j++;
        }
    }

}

int hotVektorCreater(int dictionarysize,char* file ,char** dictionary,int**hotvektor){

    FILE *dosya = fopen(file, "r");
    char karakter,temp[50];
    int kelimeIndex = 0,j=0,i,index;
    if (dosya == NULL) {
        printf("File Open Error!\n");
        return 0;
    }

    while ((karakter = fgetc(dosya)) != EOF) {
        while(karakter != '.'&& karakter != EOF){
            if(isalnum(karakter)){
                temp[kelimeIndex] = karakter;
                kelimeIndex++;
            }
            else if(isalnum(karakter)==0){
                index=findDictionarIndex(dictionary,temp,dictionarysize);
                hotvektor[index][j]=1;
                printf("[%d %d] %s %s\n",index,j,temp,dictionary[index]);
                for(int k = 0 ; k< kelimeIndex ; k++)
                {
                    temp[k] = '\0';
                }
                 kelimeIndex =0;
            }
            karakter = fgetc(dosya);
        }
        kelimeIndex = 0;
        j++;
    }
    fclose(dosya);
    return j;
}

int findDictionarIndex(char** dictionary,char* kelime,int dictionarSize){
    int i=0,lasti = 0;
     for (i = 0; kelime[i] != '\0'; i++) {
        kelime[i] = tolower(kelime[i]);
    }

    for(i=0;i<dictionarSize;i++){
        if(strcmp(dictionary[i],kelime)==0){
                lasti = i;
            return i;
        }
    }
    return lasti;
}

double tanh_activation(double x) {
    return tanh(x);
}

double tanh_derivative(double x) {
    return 1.0 - pow(tanh(x), 2);
}

void initialize_weights(double *weights) {
    srand(time(NULL));
    for (int i = 0; i < INPUT_SIZE + 1; i++) {
        weights[i] = (double)rand() / RAND_MAX; // Rastgele baþlat
    }
}

void forward_propagation(double *input, double *weights, double *output) {
    double sum = weights[0]; // bias term
    for (int i = 1; i <= INPUT_SIZE; i++) {
        sum += input[i-1] * weights[i];
    }
    *output = tanh_activation(sum);
}

void backward_propagation(double *input, double *weights, double output, double target) {
    double error = output - target;
    double delta = error * tanh_derivative(output);

    weights[0] -= LEARNING_RATE * delta; // bias term
    for (int i = 1; i <= INPUT_SIZE; i++) {
        weights[i] -= LEARNING_RATE * delta * input[i-1];
    }
}

void train(double **training_set, int training_size, double *weights) {
    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        double total_loss = 0.0;

        for (int i = 0; i < training_size; i++) {
            double input[INPUT_SIZE];
            for (int j = 0; j < INPUT_SIZE; j++) {
                input[j] = training_set[i][j];
            }

            double output;
            forward_propagation(input, weights, &output);

            double target = training_set[i][INPUT_SIZE];
            total_loss += pow(output - target, 2);

            backward_propagation(input, weights, output, target);
        }

        if (epoch % 100 == 0) {
            printf("Epoch %d, Loss: %lf\n", epoch, total_loss);
        }
    }
}
