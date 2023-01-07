Network.cs
using System;
using System.Collections.Generic;
using System.IO;

namespace Neural
{
    public class Network
    {
        private NetworkTools tools;
        public TrainSet trainSet;
        private double[][] output;
        private double[][] bias;
        private double[][] error_signal;
        private double[][] output_derivative;
        public double[][][] weights;
        public int[] NETWORK_LAYER_SIZES;
        public int INPUT_SIZE;
        public int OUTPUT_SIZE;
        public int NETWORK_SIZE;

        public Network(int[] NetworkLayers)
        {
            tools = new NetworkTools();
            NETWORK_LAYER_SIZES = NetworkLayers;
            INPUT_SIZE = NetworkLayers[0];
            OUTPUT_SIZE = NetworkLayers[NetworkLayers.Length - 1];
            NETWORK_SIZE = NetworkLayers.Length;
            trainSet = new TrainSet(INPUT_SIZE, OUTPUT_SIZE);
            output = new double[NETWORK_SIZE][];
            bias = new double[NETWORK_SIZE][];
            error_signal = new double[NETWORK_SIZE][];
            output_derivative = new double[NETWORK_SIZE][];
            weights = new double[NETWORK_SIZE][][];
            for (var i = 0; i < NETWORK_SIZE; i++)
            {
                output[i] = new double[NETWORK_LAYER_SIZES[i]];
                bias[i] = tools.createRandomArray(NETWORK_LAYER_SIZES[i], 0.3, 0.7);
                error_signal[i] = new double[NETWORK_LAYER_SIZES[i]];
                output_derivative[i] = new double[NETWORK_LAYER_SIZES[i]];
                if (i > 0)
                {
                    weights[i] = tools.createRandomArray(NETWORK_LAYER_SIZES[i], NETWORK_LAYER_SIZES[i - 1], -0.9, 0.9);
                }
            }
        }

        public Network(Network nn)
        {
            tools = nn.tools;
            trainSet = nn.trainSet;
            output = nn.output;
            bias = nn.bias;
            error_signal = nn.error_signal;
            output_derivative = nn.output_derivative;
            weights = nn.weights;
            NETWORK_LAYER_SIZES = nn.NETWORK_LAYER_SIZES;
            INPUT_SIZE = nn.INPUT_SIZE;
            OUTPUT_SIZE = nn.OUTPUT_SIZE;
            NETWORK_SIZE = nn.NETWORK_SIZE;
        }


        /// <summary>
        /// Writes the given object instance to a binary file.
        /// <para>Object type (and all child types) must be decorated with the [Serializable] attribute.</para>
        /// <para>To prevent a variable from being serialized, decorate it with the [NonSerialized] attribute; cannot be applied to properties.</para>
        /// </summary>
        /// <typeparam name="T">The type of object being written to the XML file.</typeparam>
        /// <param name="filePath">The file path to write the object instance to.</param>
        /// <param name="objectToWrite">The object instance to write to the XML file.</param>
        /// <param name="append">If false the file will be overwritten if it already exists. If true the contents will be appended to the file.</param>
        private void WriteToBinaryFile<T>(string filePath, T objectToWrite, bool append = false)
        {
            using (Stream stream = File.Open(filePath, append ? FileMode.Append : FileMode.Create))
            {
                var binaryFormatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
                binaryFormatter.Serialize(stream, objectToWrite);
            }
        }

        /// <summary>
        /// Reads an object instance from a binary file.
        /// </summary>
        /// <typeparam name="T">The type of object to read from the XML.</typeparam>
        /// <param name="filePath">The file path to read the object instance from.</param>
        /// <returns>Returns a new instance of the object read from the binary file.</returns>
        private T ReadFromBinaryFile<T>(string filePath)
        {
            using (Stream stream = File.Open(filePath, FileMode.Open))
            {
                var binaryFormatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
                return (T) binaryFormatter.Deserialize(stream);
            }
        }

        public void saveNetwork(string weights_path, string bias_path)
        {
            WriteToBinaryFile(weights_path, weights, false);
            WriteToBinaryFile(bias_path, bias, false);
        }

        public void loadNetwork(string weights_path, string bias_path)
        {
            weights = ReadFromBinaryFile<double[][][]>(weights_path);
            bias = ReadFromBinaryFile<double[][]>(bias_path);
        }

        private double sigmoid(double value) => 1.0 / (1 + Math.Exp(-value));

        public double[] calculate(double[] input)
        {
            if (input.Length != INPUT_SIZE) return null;
            output[0] = input;
            for (int layer = 1; layer < NETWORK_SIZE; layer++)
            {
                for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++)
                {
                    var sum = bias[layer][neuron];
                    for (int prevL = 0; prevL < NETWORK_LAYER_SIZES[layer - 1]; prevL++)
                    {
                        sum += output[layer - 1][prevL] * weights[layer][neuron][prevL];
                    }

                    output[layer][neuron] = sigmoid(sum);
                    output_derivative[layer][neuron] = (sigmoid(sum) * (1 - sigmoid(sum)));
                }
            }

            return output[NETWORK_SIZE - 1];
        }

        public void train(int loops, int batch_size, double learningRate)
        {
            if (trainSet.INPUT_SIZE != INPUT_SIZE || trainSet.OUTPUT_SIZE != OUTPUT_SIZE) return;
            for (int i = 0; i < loops; i++)
            {
                TrainSet batch = trainSet.extractBatch(batch_size);
                for (int b = 0; b < batch.size(); b++)
                {
                    train(batch.getInput(b), batch.getOutput(b), learningRate);
                }
            }
        }

        public void train(double[] input, double[] target, double learningRate)
        {
            if (input.Length != INPUT_SIZE || target.Length != OUTPUT_SIZE) return;
            calculate(input);
            backpropError(target);
            updateWeights(learningRate);
        }

        public void backpropError(double[] target)
        {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[NETWORK_LAYER_SIZES.Length - 1]; neuron++)
            {
                error_signal[NETWORK_SIZE - 1][neuron] = (output[NETWORK_SIZE - 1][neuron] - target[neuron]) *
                                                         output_derivative[NETWORK_SIZE - 1][neuron];
            }

            for (int layer = NETWORK_SIZE - 2; layer > 0; layer--)
            {
                for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++)
                {
                    var sum = 0.0;
                    for (int nextNeuron = 0; nextNeuron < NETWORK_LAYER_SIZES[layer + 1]; nextNeuron++)
                    {
                        sum += weights[layer + 1][nextNeuron][neuron] * error_signal[layer + 1][nextNeuron];
                    }

                    error_signal[layer][neuron] = sum * output_derivative[layer][neuron];
                }
            }
        }

        public void updateWeights(double eta)
        {
            for (int layer = 1; layer < NETWORK_SIZE; layer++)
            {
                for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++)
                {
                    double delta;
                    for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++)
                    {
                        delta = -eta * output[layer - 1][prevNeuron] * error_signal[layer][neuron];
                        weights[layer][neuron][prevNeuron] += delta;
                    }

                    delta = -eta * error_signal[layer][neuron];
                    bias[layer][neuron] += delta;
                }
            }
        }

        public double MSE(double[] input, double[] target)
        {
            if (input.Length != INPUT_SIZE || target.Length != OUTPUT_SIZE) return -1;
            calculate(input);
            double v = 0;
            for (int i = 0; i < target.Length; i++)
            {
                v += Math.Pow(target[i] - output[NETWORK_SIZE - 1][i], 2);
            }

            return v / (2d * target.Length);
        }

        public double MSE()
        {
            double v = 0;
            for (int i = 0; i < trainSet.size(); i++)
            {
                v += MSE(trainSet.getInput(i), trainSet.getOutput(i));
            }

            return v / (2d * trainSet.size());
        }

        public void mutate(double probability = 0.2)
        {
            tools.ApplyMutation(ref weights, probability);
            tools.ApplyMutation(ref bias, probability);
        }

        public Network merge(Network nn, double probability = 0.5)
        {
            if (this.INPUT_SIZE != nn.INPUT_SIZE || this.OUTPUT_SIZE != nn.OUTPUT_SIZE)
            {
                throw new Exception("Networks not of the same size!");
            }

            Network result = this.copy();

            for (int layer = 0; layer < weights.GetLength(0); layer++)
            {
                for (int neuron = 0; neuron < weights.GetLength(1); neuron++)
                {
                    for (int single = 0; single < weights.GetLength(2); single++)
                    {
                        double chance = tools.randomValue(0, 1);
                        if (chance >= probability)
                        {
                            result.weights[layer][neuron][single] = nn.weights[layer][neuron][single];
                        }
                    }

                    double chanceBias = tools.randomValue(0, 1);
                    if (chanceBias >= probability)
                    {
                        result.bias[layer][neuron] = nn.bias[layer][neuron];
                    }
                }
            }

            return result;
        }

        public Network copy()
        {
            return new Network(this);
        }

        public class TrainSet
        {
            struct dataStruct
            {
                public double[] input;
                public double[] output;
            }

            public int INPUT_SIZE;
            public int OUTPUT_SIZE;
            private NetworkTools tools;

            private List<dataStruct> allData;

            public TrainSet(int INPUT_SIZE, int OUTPUT_SIZE)
            {
                this.INPUT_SIZE = INPUT_SIZE;
                this.OUTPUT_SIZE = OUTPUT_SIZE;
                tools = new NetworkTools();
                allData = new List<dataStruct>();
            }

            public void addData(double[] dataIn, double[] expected)
            {
                if (dataIn.Length != INPUT_SIZE || expected.Length != OUTPUT_SIZE) return;
                allData.Add(new dataStruct {input = dataIn, output = expected});
            }

            public void clearData()
            {
                allData = new List<dataStruct>();
            }

            public TrainSet extractBatch(int size)
            {
                if (size <= 0 || size > this.size()) return this;
                var set = new TrainSet(INPUT_SIZE, OUTPUT_SIZE);
                var ids = tools.randomValues(0, this.size(), size);
                foreach (var temp in ids)
                {
                    set.addData(getInput(temp), getOutput(temp));
                }

                return set;
            }

            public int size()
            {
                return allData.Count;
            }

            public double[] getInput(int index)
            {
                if (index >= 0 && index < size())
                    return allData[index].input;
                return null;
            }

            public double[] getOutput(int index)
            {
                if (index >= 0 && index < size())
                    return allData[index].output;
                return null;
            }

            public int getINPUT_SIZE()
            {
                return INPUT_SIZE;
            }

            public int getOUTPUT_SIZE()
            {
                return OUTPUT_SIZE;
            }
        }

        public class NetworkTools
        {
            private Random rand;

            public NetworkTools()
            {
                rand = new Random(Guid.NewGuid().GetHashCode());
            }
            
            public NetworkTools(int seed)
            {
                rand = new Random(seed);
            }

            public double[] createArray(int size, double init_value)
            {
                if (size < 1) return null;
                var ar = new double[size];
                for (var i = 0; i < ar.Length; i++)
                {
                    ar[i] = init_value;
                }

                return ar;
            }

            public void ApplyMutation(ref double[][][] weights, double probability)
            {
                for (int layer = 1; layer < weights.GetLength(0); layer++)
                {
                    for (int neuron = 0; neuron < weights[layer].Length; neuron++)
                    {
                        for (int single = 0; single < weights[layer][neuron].Length; single++)
                        {
                            double chance = randomValue(0, 1);
                            if (probability >= chance)
                            {
                                double value = randomValue(-1, 1);
                                weights[layer][neuron][single] = value;
                            }
                        }
                    }
                }
            }

            public void ApplyMutation(ref double[][] bias, double probability)
            {
                for (int layer = 1; layer < bias.GetLength(0); layer++)
                {
                    for (int neuron = 0; neuron < bias[layer].Length; neuron++)
                    {
                        double chance = randomValue(0, 1);
                        if (probability >= chance)
                        {
                            double value = randomValue(-1, 1);
                            bias[layer][neuron] = value;
                        }
                    }
                }
            }

            public double[] createRandomArray(int size, double lower_bound, double upper_bound)
            {
                if (size < 1) return null;
                var ar = new double[size];
                for (var i = 0; i < ar.Length; i++)
                {
                    ar[i] = randomValue(lower_bound, upper_bound);
                }

                return ar;
            }

            public double[][] createRandomArray(int sizeX, int sizeY, double lower_bound, double upper_bound)
            {
                if (sizeX < 1 || sizeY < 1) return null;
                var ar = new double[sizeX][];
                for (var i = 0; i < sizeX; i++)
                {
                    ar[i] = createRandomArray(sizeY, lower_bound, upper_bound);
                }

                return ar;
            }

            public double randomValue(double lower_bound, double upper_bound)
            {
                return rand.NextDouble() * (upper_bound - lower_bound) + lower_bound;
            }

            private bool contais(int[] values, int value)
            {
                foreach (var i in values)
                {
                    if (i == -9) return false;
                    if (i == value) return true;
                }

                return false;
            }

            public int[] randomValues(int lowerBound, int upperBound, int amount)
            {
                lowerBound--;

                if (amount > (upperBound - lowerBound)) return null;

                var values = new int[amount];
                for (var i = 0; i < values.Length; i++) values[i] = -9;
                for (var n = 0; n < values.Length; n++)
                {
                    var i = rand.Next(lowerBound + 1, upperBound);
                    while (contais(values, i))
                    {
                        i = rand.Next(lowerBound + 1, upperBound);
                    }

                    values[n] = i;
                }

                return values;
            }

            public static int indexOfHighestValue(double[] values)
            {
                int index = 0;
                for (int i = 1; i < values.Length; i++)
                {
                    if (values[i] > values[index])
                    {
                        index = i;
                    }
                }

                return index;
            }
            
            public static int indexOfHighestValue(double[] values, int from, int to)
            {
                int index = from;
                for (int i = from; i < to; i++)
                {
                    if (values[i] > values[index])
                    {
                        index = i;
                    }
                }

                return index;
            }
        }
    }
}