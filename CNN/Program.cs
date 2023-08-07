using CNN;
using CNN.Utils;

int nrSamples = 500;
int Nx = 28;
int Ny = 28;
var inputData = LoadData.LoadInputData("TrainingData\\images.csv", nrSamples, Nx, Ny, ',', 1.0 / 256.0);
var outputData = LoadData.LoadOutputData("TrainingData\\numbers.csv", nrSamples);

int nrEpochs = 1;
int batchSize = 10;
int nrTrainingSteps = 100;
double learningRate = 0.01;
double dropout = 0.0;

ConvolutionNeuralNet cnet = new ConvolutionNeuralNet();
cnet.AddRandomConvolutionLayer(5, 5, 1, 1, 2);
cnet.AddPoolingLayer(2, 2, 2, 2);
cnet.AddRandomConvolutionLayer(3, 3, 1, 2, 4);
cnet.AddPoolingLayer(2, 2, 2, 4);

int nrInputNodes = 100;
int nrOutputNodes = 10;
cnet.AddRandomNeuralNetwork(batchSize, new int[] { nrInputNodes, nrOutputNodes }, "ReLU", "Softmax", dropout);

int nrTrainingImages = 300;
int nrValidationImages = 30;
cnet.Train(inputData, outputData, 0, nrTrainingImages, learningRate, nrTrainingSteps, batchSize, nrEpochs);
cnet.Infer(inputData, outputData, nrTrainingImages, nrTrainingImages + nrValidationImages);









