using CNN.Utils;

namespace CNN
{
    /// <summary>
    /// Implements a fully connected neural network with an arbitrary number of hidden layers.
    /// 
    /// </summary>
    public class NeuralNetwork
    {
        public List<NeuralNetLayer> layers = new List<NeuralNetLayer>();

        public int NrLayers { get { return layers.Count; } }

        public NeuralNetwork()
        {

        }

        /// <summary>
        /// Randomly initialize the weight matrix for each layer with He initilaization.
        /// </summary>
        /// <param name="batchSize"></param>
        /// <param name="layerSizes"></param>
        /// <param name="activationFunction"></param>
        /// <param name="outputLayerActivationFunction"></param>
        /// <param name="dropoutRate"></param>
        /// <returns></returns>
        public static NeuralNetwork GenerateRandomNeuralNetwork(int batchSize, int[] layerSizes, string activationFunction,
            string outputLayerActivationFunction, double dropoutRate)
        {
            NeuralNetwork nn = new NeuralNetwork();

            nn.layers.Add(new NeuralNetLayer(batchSize, null, layerSizes[0], activationFunction, dropoutRate));
            for (int i = 1; i < layerSizes.Length - 1; i++)
            {
                nn.layers.Add(new NeuralNetLayer(batchSize, nn.layers[i - 1], layerSizes[i], activationFunction, dropoutRate));
                nn.layers[i].CreateRandomParameters();
            }
            nn.layers.Add(new NeuralNetLayer(batchSize, nn.layers[layerSizes.Length - 2], layerSizes[layerSizes.Length - 1], outputLayerActivationFunction, 0));
            nn.layers[layerSizes.Length - 1].CreateRandomParameters();

            return nn;
        }

        /// <summary>
        /// Forward pass of network.
        /// </summary>
        /// <param name="inputValues"></param>
        /// <param name="isTraining"></param>
        public void Evaluate(Tensor inputValues, bool isTraining)
        {
            int batchSize = inputValues.GetLength(0);
            int layerSize = inputValues.GetLength(1);
            layers[0].SetInputBatchSize(batchSize);
            for (int batch = 0; batch < batchSize; batch++)
            {
                for (int i = 0; i < layerSize; i++)
                {
                    layers[0].SetInputValue(batch, i, inputValues[batch, i]);
                }
            }

            for (int i = 0; i < layers.Count; i++)
            {
                layers[i].Update(isTraining);
            }

        }

        /// <summary>
        /// Calculate the cross entropy between the predicted output digit and the actual value.
        /// </summary>
        /// <param name="observations">Actual values read from training data text file.</param>
        public Rev EvaluateLossFunction(double[] observations, bool isTraining)
        {
            Rev loss = 0;

            if (isTraining)
            {
                for (int batch = 0; batch < layers[NrLayers - 1].BatchSize; batch++)
                {
                    for (int i = 0; i < layers[NrLayers - 1].LayerSize; i++)
                    {
                        if (observations[batch] == i)
                            loss -= layers[NrLayers - 1].GetOutputValue(batch, i).Log();
                    }
                }
                loss = loss / (double)(layers[NrLayers - 1].BatchSize);
            }
            else
            {
                for (int batch = 0; batch < layers[NrLayers - 1].BatchSize; batch++)
                {
                    string s = observations[batch].ToString() + ": ";
                    double max = 0;
                    int maxind = 0;
                    for (int i = 0; i < layers[NrLayers - 1].LayerSize; i++)
                    {
                        if (layers[NrLayers - 1].GetOutputValue(batch, i).Magnitude > max)
                        {
                            max = layers[NrLayers - 1].GetOutputValue(batch, i).Magnitude;
                            maxind = i;
                        }

                        if (observations[batch] == i)
                            loss -= layers[NrLayers - 1].GetOutputValue(batch, i).Log();
                    }
                    s += maxind.ToString() + ", " + (100 * layers[NrLayers - 1].GetOutputValue(batch, maxind).Magnitude).ToString("F3") + " %";
                    Console.WriteLine(s);
                }
                loss = loss / (double)(layers[NrLayers - 1].BatchSize);
            }
            return loss;
        }

        /// <summary>
        /// Backward pass of network.
        /// </summary>
        public Rev CalculateLossFunctionDerivatives(double[] observations)
        {
            Rev loss = EvaluateLossFunction(observations, false);
            loss.CalculateDerivative(1);
            return loss;
        }

        public void MakeTrainingStep(double learningRate, int step)
        {
            for (int i = 0; i < layers.Count; i++)
            {
                layers[i].MakeTrainingStep(learningRate, step);
            }
        }

    }
}
