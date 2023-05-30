using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Neural
{
    public class Network
    {
        public enum ActivationFunction
        {
            Sigmoid,
            Tanh,
            LeakyRelu
        }
        private NetworkTools tools;
        public TrainSet trainSet;
        private ActivationFunction activationFunction;
        private IActivationMethods activationMethod;
        private double[][] output;
        private double[][] bias;
        private double[][] error_signal;
        private double[][] output_derivative;
        public double[][][] weights;
        public int[] networkLayerSizes;
        public int inputSize;
        public int outputSize;
        public int networkSize;

        public Network(int[] NetworkLayers, ActivationFunction activationFunction = ActivationFunction.Tanh)
        {
            tools = new NetworkTools();
            this.activationFunction = activationFunction;
            networkLayerSizes = NetworkLayers;
            inputSize = NetworkLayers[0];
            outputSize = NetworkLayers[NetworkLayers.Length - 1];
            networkSize = NetworkLayers.Length;
            trainSet = new TrainSet(inputSize, outputSize);
            output = new double[networkSize][];
            bias = new double[networkSize][];
            error_signal = new double[networkSize][];
            output_derivative = new double[networkSize][];
            weights = new double[networkSize][][];
            for (var i = 0; i < networkSize; i++)
            {
                output[i] = new double[networkLayerSizes[i]];
                bias[i] = tools.CreateRandomArray(networkLayerSizes[i], 0.3, 0.7);
                error_signal[i] = new double[networkLayerSizes[i]];
                output_derivative[i] = new double[networkLayerSizes[i]];
                if (i > 0)
                {
                    weights[i] = tools.CreateRandomArray(networkLayerSizes[i], networkLayerSizes[i - 1], -0.9, 0.9);
                }
            }

            SetActivationMethod();
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
            networkLayerSizes = nn.networkLayerSizes;
            inputSize = nn.inputSize;
            outputSize = nn.outputSize;
            networkSize = nn.networkSize;
            activationFunction = nn.activationFunction;
            SetActivationMethod();
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
#pragma warning disable SYSLIB0011 // Type or member is obsolete
                binaryFormatter.Serialize(stream, objectToWrite);
#pragma warning restore SYSLIB0011 // Type or member is obsolete
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
#pragma warning disable SYSLIB0011 // Type or member is obsolete
                return (T)binaryFormatter.Deserialize(stream);
#pragma warning restore SYSLIB0011 // Type or member is obsolete
            }
        }

        public void SaveNetwork(string weights_path, string bias_path)
        {
            WriteToBinaryFile(weights_path, weights, false);
            WriteToBinaryFile(bias_path, bias, false);
        }

        public void LoadNetwork(string weights_path, string bias_path)
        {
            weights = ReadFromBinaryFile<double[][][]>(weights_path);
            bias = ReadFromBinaryFile<double[][]>(bias_path);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public double[] Calculate(ref double[] input)
        {
            if (input.Length != inputSize) return null;
            output[0] = input;
            for (int layer = 1; layer < networkSize; layer++)
            {
                for (int neuron = 0; neuron < networkLayerSizes[layer]; neuron++)
                {
                    var sum = bias[layer][neuron];
                    for (int prevL = 0; prevL < networkLayerSizes[layer - 1]; prevL++)
                    {
                        sum += output[layer - 1][prevL] * weights[layer][neuron][prevL];
                    }

                    output[layer][neuron] = activationMethod.Activation(sum);
                    output_derivative[layer][neuron] = activationMethod.Deactivation(sum);
                }
            }

            return output[networkSize - 1];
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Train(int loops, int batch_size, double learningRate)
        {
            if (trainSet.inputSize != inputSize || trainSet.outputSize != outputSize) return;
            for (int i = 0; i < loops; i++)
            {
                TrainSet batch = trainSet.ExtractBatch(batch_size);
                for (int b = 0; b < batch.Size(); b++)
                {
                    var input = batch.GetInput(b);
                    var output = batch.GetOutput(b);
                    Train(ref input, ref output, learningRate);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void TrainFor(TimeSpan timeSpan, int batch_size, double learningRate)
        {
            if (trainSet.inputSize != inputSize || trainSet.outputSize != outputSize) return;
            DateTime endTime = DateTime.Now + timeSpan;
            while (DateTime.Now < endTime)
            {
                TrainSet batch = trainSet.ExtractBatch(batch_size);
                for (int b = 0; b < batch.Size(); b++)
                {
                    var input = batch.GetInput(b);
                    var output = batch.GetOutput(b);
                    Train(ref input, ref output, learningRate);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Train(ref double[] input, ref double[] target, double learningRate)
        {
            if (input.Length != inputSize || target.Length != outputSize) return;
            Calculate(ref input);
            backpropError(ref target);
            UpdateWeights(learningRate);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void backpropError(ref double[] target)
        {
            Span<int> spanNetworkSizes = networkLayerSizes;
            ref int networkSizeBuffer = ref MemoryMarshal.GetReference(spanNetworkSizes);

            Span<double> spanTarget = target;
            ref double targetBuffer = ref MemoryMarshal.GetReference(spanTarget);

            Span<double[]> errorSignal = error_signal;
            Span<double[]> outputS = output;
            Span<double[]> outputDerivative = output_derivative;
            for (int neuron = 0; neuron < spanNetworkSizes[spanNetworkSizes.Length - 1]; neuron++)
            {
                double targetNeuron = Unsafe.Add(ref targetBuffer, neuron);
                errorSignal[networkSize - 1][neuron] = (outputS[networkSize - 1][neuron] - targetNeuron) *
                                                         outputDerivative[networkSize - 1][neuron];
            }

            Span<double[][]> weightTops = weights;
            for (int layer = networkSize - 2; layer > 0; layer--)
            {
                int firstLayerSize = Unsafe.Add(ref networkSizeBuffer, layer);
                int secondLayerSize = Unsafe.Add(ref networkSizeBuffer, layer + 1);
                for (int neuron = 0; neuron < firstLayerSize; neuron++)
                {
                    var sum = 0.0;
                    for (int nextNeuron = 0; nextNeuron < secondLayerSize; nextNeuron++)
                    {
                        sum += weightTops[layer + 1][nextNeuron][neuron] * errorSignal[layer + 1][nextNeuron];
                    }

                    errorSignal[layer][neuron] = sum * outputDerivative[layer][neuron];
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void UpdateWeights(double eta)
        {
            var networkLayerSizesSpan = networkLayerSizes.AsSpan();
            ref int networkSizeBuffer = ref MemoryMarshal.GetReference(networkLayerSizesSpan);

            var weightsSpan = weights.AsSpan();
            var errorSignalSpan = error_signal.AsSpan();
            var biasSpan = bias.AsSpan();
            var outputSpan = output.AsSpan();
            for (int layer = 1; layer < networkSize; layer++)
            {
                int layerSizeLast = Unsafe.Add(ref networkSizeBuffer, layer - 1);
                int layerSize = Unsafe.Add(ref networkSizeBuffer, layer);
                for (int neuron = 0; neuron < layerSize; neuron++)
                {
                    double delta;
                    for (int prevNeuron = 0; prevNeuron < layerSizeLast; prevNeuron++)
                    {
                        delta = -eta * outputSpan[layer - 1][prevNeuron] * errorSignalSpan[layer][neuron];
                        weightsSpan[layer][neuron][prevNeuron] += delta;
                    }

                    delta = -eta * errorSignalSpan[layer][neuron];
                    biasSpan[layer][neuron] += delta;
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public double MSE(ref double[] input, ref double[] target)
        {
            if (input.Length != inputSize || target.Length != outputSize) return -1;
            Calculate(ref input);
            double v = 0;
            for (int i = 0; i < target.Length; i++)
            {
                v += Math.Pow(target[i] - output[networkSize - 1][i], 2);
            }

            return v / (2d * target.Length);
        }

        public double MSE()
        {
            double v = 0;
            for (int i = 0; i < trainSet.Size(); i++)
            {
                var input = trainSet.GetInput(i);
                var output = trainSet.GetOutput(i);
                v += MSE(ref input, ref output);
            }

            return v / (2d * trainSet.Size());
        }

        public void Mutate(double probability = 0.2)
        {
            tools.ApplyMutation(ref weights, probability);
            tools.ApplyMutation(ref bias, probability);
        }

        public Network Merge(Network nn, double probability = 0.5)
        {
            if (this.inputSize != nn.inputSize || this.outputSize != nn.outputSize)
            {
                throw new Exception("Networks not of the same size!");
            }

            Network result = this.Copy();

            for (int layer = 0; layer < weights.GetLength(0); layer++)
            {
                for (int neuron = 0; neuron < weights.GetLength(1); neuron++)
                {
                    for (int single = 0; single < weights.GetLength(2); single++)
                    {
                        double chance = tools.RandomValue(0, 1);
                        if (chance >= probability)
                        {
                            result.weights[layer][neuron][single] = nn.weights[layer][neuron][single];
                        }
                    }

                    double chanceBias = tools.RandomValue(0, 1);
                    if (chanceBias >= probability)
                    {
                        result.bias[layer][neuron] = nn.bias[layer][neuron];
                    }
                }
            }

            return result;
        }

        public Network Copy()
        {
            return new Network(this);
        }

        #region Activation methods
        private void SetActivationMethod()
        {
            if (this.activationFunction == ActivationFunction.Sigmoid)
            {
                activationMethod = new ActivationSigmoid();
            }
            else if (this.activationFunction == ActivationFunction.Tanh)
            {
                activationMethod = new ActivationTanh();
            }
            else if (this.activationFunction == ActivationFunction.LeakyRelu)
            {
                activationMethod = new ActivationLeakyRelu();
            }
        }

        private interface IActivationMethods
        {
            public double Activation(double value);
            public double Deactivation(double value);
        }

        private class ActivationSigmoid : IActivationMethods
        {
            public double Activation(double value)
            {
                return 1.0 / (1.0 + Math.Exp(-value));
            }

            public double Deactivation(double value)
            {
                return Activation(value) * (1 - Activation(value));
            }
        }

        private class ActivationTanh : IActivationMethods
        {
            public double Activation(double value)
            {
                return Math.Tanh(value);
            }

            public double Deactivation(double value)
            {
                return 1.0 - Math.Pow(Math.Tanh(value), 2.0);
            }
        }

        private class ActivationLeakyRelu : IActivationMethods
        {
            public double Activation(double value)
            {
                return Math.Max(0.1 * value, value);
            }

            public double Deactivation(double value)
            {
                return value >= 0 ? 1 : 0.01;
            }
        }
        #endregion

        public class TrainSet
        {
            struct dataStruct
            {
                public double[] input;
                public double[] output;
            }

            public int inputSize;
            public int outputSize;
            private NetworkTools tools;

            private List<dataStruct> allData;

            public TrainSet(int INPUT_SIZE, int OUTPUT_SIZE)
            {
                this.inputSize = INPUT_SIZE;
                this.outputSize = OUTPUT_SIZE;
                tools = new NetworkTools();
                allData = new List<dataStruct>();
            }

            public void AddData(double[] dataIn, double[] expected)
            {
                if (dataIn.Length != inputSize || expected.Length != outputSize) return;
                allData.Add(new dataStruct { input = dataIn, output = expected });
            }

            public void ClearData()
            {
                allData = new List<dataStruct>();
            }

            // Fisher-Yates shuffle algorithm
            public void ShuffleData()
            {
                Random rand = new();
                int n = allData.Count;
                while (n > 1)
                {
                    n--;
                    int k = rand.Next(n + 1);
                    var value = allData[k];
                    allData[k] = allData[n];
                    allData[n] = value;
                }
            }

            public TrainSet ExtractBatch(int size)
            {
                if (size <= 0 || size > this.Size()) return this;
                var set = new TrainSet(inputSize, outputSize);
                var ids = tools.RandomValues(0, this.Size(), size);
                foreach (var temp in ids)
                {
                    set.AddData(GetInput(temp), GetOutput(temp));
                }

                return set;
            }

            public int Size()
            {
                return allData.Count;
            }

            public double[] GetInput(int index)
            {
                if (index >= 0 && index < Size())
                    return allData[index].input;
                return null;
            }

            public double[] GetOutput(int index)
            {
                if (index >= 0 && index < Size())
                    return allData[index].output;
                return null;
            }

            public int GetInputSize()
            {
                return inputSize;
            }

            public int GetOutputSize()
            {
                return outputSize;
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

            public void ApplyMutation(ref double[][][] weights, double probability)
            {
                for (int layer = 1; layer < weights.GetLength(0); layer++)
                {
                    for (int neuron = 0; neuron < weights[layer].Length; neuron++)
                    {
                        for (int single = 0; single < weights[layer][neuron].Length; single++)
                        {
                            double chance = RandomValue(0, 1);
                            if (probability >= chance)
                            {
                                double value = RandomValue(-1, 1);
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
                        double chance = RandomValue(0, 1);
                        if (probability >= chance)
                        {
                            double value = RandomValue(-1, 1);
                            bias[layer][neuron] = value;
                        }
                    }
                }
            }

            public double[] CreateRandomArray(int size, double lower_bound, double upper_bound)
            {
                if (size < 1) return null;
                var ar = new double[size];
                for (var i = 0; i < ar.Length; i++)
                {
                    ar[i] = RandomValue(lower_bound, upper_bound);
                }

                return ar;
            }

            public double[][] CreateRandomArray(int sizeX, int sizeY, double lower_bound, double upper_bound)
            {
                if (sizeX < 1 || sizeY < 1) return null;
                var ar = new double[sizeX][];
                for (var i = 0; i < sizeX; i++)
                {
                    ar[i] = CreateRandomArray(sizeY, lower_bound, upper_bound);
                }

                return ar;
            }

            public double RandomValue(double lower_bound, double upper_bound)
            {
                return rand.NextDouble() * (upper_bound - lower_bound) + lower_bound;
            }

            private bool Contais(int[] values, int value)
            {
                foreach (var i in values)
                {
                    if (i == value) return true;
                }
public class Network
    {
        public enum ActivationFunction
        {
            Sigmoid,
            Tanh,
            LeakyRelu
        }
        private NetworkTools tools;
        public TrainSet trainSet;
        private ActivationFunction activationFunction;
        private IActivationMethods activationMethod;
        private double[][] output;
        private double[][] bias;
        private double[][] error_signal;
        private double[][] output_derivative;
        public double[][][] weights;
        public int[] networkLayerSizes;
        public int inputSize;
        public int outputSize;
        public int networkSize;

        public Network(int[] NetworkLayers, ActivationFunction activationFunction = ActivationFunction.Tanh)
        {
            tools = new NetworkTools();
            this.activationFunction = activationFunction;
            networkLayerSizes = NetworkLayers;
            inputSize = NetworkLayers[0];
            outputSize = NetworkLayers[NetworkLayers.Length - 1];
            networkSize = NetworkLayers.Length;
            trainSet = new TrainSet(inputSize, outputSize);
            output = new double[networkSize][];
            bias = new double[networkSize][];
            error_signal = new double[networkSize][];
            output_derivative = new double[networkSize][];
            weights = new double[networkSize][][];
            for (var i = 0; i < networkSize; i++)
            {
                output[i] = new double[networkLayerSizes[i]];
                bias[i] = tools.CreateRandomArray(networkLayerSizes[i], 0.3, 0.7);
                error_signal[i] = new double[networkLayerSizes[i]];
                output_derivative[i] = new double[networkLayerSizes[i]];
                if (i > 0)
                {
                    weights[i] = tools.CreateRandomArray(networkLayerSizes[i], networkLayerSizes[i - 1], -0.9, 0.9);
                }
            }

            SetActivationMethod();
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
            networkLayerSizes = nn.networkLayerSizes;
            inputSize = nn.inputSize;
            outputSize = nn.outputSize;
            networkSize = nn.networkSize;
            activationFunction = nn.activationFunction;
            SetActivationMethod();
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
#pragma warning disable SYSLIB0011 // Type or member is obsolete
                binaryFormatter.Serialize(stream, objectToWrite);
#pragma warning restore SYSLIB0011 // Type or member is obsolete
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
#pragma warning disable SYSLIB0011 // Type or member is obsolete
                return (T)binaryFormatter.Deserialize(stream);
#pragma warning restore SYSLIB0011 // Type or member is obsolete
            }
        }

        public void SaveNetwork(string weights_path, string bias_path)
        {
            WriteToBinaryFile(weights_path, weights, false);
            WriteToBinaryFile(bias_path, bias, false);
        }

        public void LoadNetwork(string weights_path, string bias_path)
        {
            weights = ReadFromBinaryFile<double[][][]>(weights_path);
            bias = ReadFromBinaryFile<double[][]>(bias_path);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public double[] Calculate(ref double[] input)
        {
            if (input.Length != inputSize) return null;
            output[0] = input;
            for (int layer = 1; layer < networkSize; layer++)
            {
                for (int neuron = 0; neuron < networkLayerSizes[layer]; neuron++)
                {
                    var sum = bias[layer][neuron];
                    for (int prevL = 0; prevL < networkLayerSizes[layer - 1]; prevL++)
                    {
                        sum += output[layer - 1][prevL] * weights[layer][neuron][prevL];
                    }

                    output[layer][neuron] = activationMethod.Activation(sum);
                    output_derivative[layer][neuron] = activationMethod.Deactivation(sum);
                }
            }

            return output[networkSize - 1];
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Train(int loops, int batch_size, double learningRate)
        {
            if (trainSet.inputSize != inputSize || trainSet.outputSize != outputSize) return;
            for (int i = 0; i < loops; i++)
            {
                TrainSet batch = trainSet.ExtractBatch(batch_size);
                for (int b = 0; b < batch.Size(); b++)
                {
                    var input = batch.GetInput(b);
                    var output = batch.GetOutput(b);
                    Train(ref input, ref output, learningRate);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void TrainFor(TimeSpan timeSpan, int batch_size, double learningRate)
        {
            if (trainSet.inputSize != inputSize || trainSet.outputSize != outputSize) return;
            DateTime endTime = DateTime.Now + timeSpan;
            while (DateTime.Now < endTime)
            {
                TrainSet batch = trainSet.ExtractBatch(batch_size);
                for (int b = 0; b < batch.Size(); b++)
                {
                    var input = batch.GetInput(b);
                    var output = batch.GetOutput(b);
                    Train(ref input, ref output, learningRate);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Train(ref double[] input, ref double[] target, double learningRate)
        {
            if (input.Length != inputSize || target.Length != outputSize) return;
            Calculate(ref input);
            backpropError(ref target);
            UpdateWeights(learningRate);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void backpropError(ref double[] target)
        {
            Span<int> spanNetworkSizes = networkLayerSizes;
            Span<double> spanTarget = target;

            Span<double[]> errorSignal = error_signal;
            Span<double[]> outputS = output;
            Span<double[]> outputDerivative = output_derivative;
            for (int neuron = 0; neuron < spanNetworkSizes[spanNetworkSizes.Length - 1]; neuron++)
            {
                errorSignal[networkSize - 1][neuron] = (outputS[networkSize - 1][neuron] - spanTarget[neuron]) *
                                                         outputDerivative[networkSize - 1][neuron];
            }

            Span<double[][]> weightTops = weights;
            for (int layer = networkSize - 2; layer > 0; layer--)
            {
                int firstLayerSize = spanNetworkSizes[layer];
                int secondLayerSize = spanNetworkSizes[layer + 1];
                for (int neuron = 0; neuron < firstLayerSize; neuron++)
                {
                    var sum = 0.0;
                    for (int nextNeuron = 0; nextNeuron < secondLayerSize; nextNeuron++)
                    {
                        sum += weightTops[layer + 1][nextNeuron][neuron] * errorSignal[layer + 1][nextNeuron];
                    }

                    errorSignal[layer][neuron] = sum * outputDerivative[layer][neuron];
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void UpdateWeights(double eta)
        {
            var networkLayerSizesSpan = networkLayerSizes.AsSpan();
            var weightsSpan = weights.AsSpan();
            var errorSignalSpan = error_signal.AsSpan();
            var biasSpan = bias.AsSpan();
            var outputSpan = output.AsSpan();
            for (int layer = 1; layer < networkSize; layer++)
            {
                int layerSizeLast = networkLayerSizesSpan[layer - 1];
                int layerSize = networkLayerSizesSpan[layer];
                for (int neuron = 0; neuron < layerSize; neuron++)
                {
                    double delta;
                    for (int prevNeuron = 0; prevNeuron < layerSizeLast; prevNeuron++)
                    {
                        delta = -eta * outputSpan[layer - 1][prevNeuron] * errorSignalSpan[layer][neuron];
                        weightsSpan[layer][neuron][prevNeuron] += delta;
                    }

                    delta = -eta * errorSignalSpan[layer][neuron];
                    biasSpan[layer][neuron] += delta;
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public double MSE(ref double[] input, ref double[] target)
        {
            if (input.Length != inputSize || target.Length != outputSize) return -1;
            Calculate(ref input);
            double v = 0;
            for (int i = 0; i < target.Length; i++)
            {
                v += Math.Pow(target[i] - output[networkSize - 1][i], 2);
            }

            return v / (2d * target.Length);
        }

        public double MSE()
        {
            double v = 0;
            for (int i = 0; i < trainSet.Size(); i++)
            {
                var input = trainSet.GetInput(i);
                var output = trainSet.GetOutput(i);
                v += MSE(ref input, ref output);
            }

            return v / (2d * trainSet.Size());
        }

        public void Mutate(double probability = 0.2)
        {
            tools.ApplyMutation(ref weights, probability);
            tools.ApplyMutation(ref bias, probability);
        }

        public Network Merge(Network nn, double probability = 0.5)
        {
            if (this.inputSize != nn.inputSize || this.outputSize != nn.outputSize)
            {
                throw new Exception("Networks not of the same size!");
            }

            Network result = this.Copy();

            for (int layer = 0; layer < weights.GetLength(0); layer++)
            {
                for (int neuron = 0; neuron < weights.GetLength(1); neuron++)
                {
                    for (int single = 0; single < weights.GetLength(2); single++)
                    {
                        double chance = tools.RandomValue(0, 1);
                        if (chance >= probability)
                        {
                            result.weights[layer][neuron][single] = nn.weights[layer][neuron][single];
                        }
                    }

                    double chanceBias = tools.RandomValue(0, 1);
                    if (chanceBias >= probability)
                    {
                        result.bias[layer][neuron] = nn.bias[layer][neuron];
                    }
                }
            }

            return result;
        }

        public Network Copy()
        {
            return new Network(this);
        }

        #region Activation methods
        private void SetActivationMethod()
        {
            if (this.activationFunction == ActivationFunction.Sigmoid)
            {
                activationMethod = new ActivationSigmoid();
            }
            else if (this.activationFunction == ActivationFunction.Tanh)
            {
                activationMethod = new ActivationTanh();
            }
            else if (this.activationFunction == ActivationFunction.LeakyRelu)
            {
                activationMethod = new ActivationLeakyRelu();
            }
        }

        private interface IActivationMethods
        {
            public double Activation(double value);
            public double Deactivation(double value);
        }

        private class ActivationSigmoid : IActivationMethods
        {
            public double Activation(double value)
            {
                return 1.0 / (1.0 + Math.Exp(-value));
            }

            public double Deactivation(double value)
            {
                return Activation(value) * (1 - Activation(value));
            }
        }

        private class ActivationTanh : IActivationMethods
        {
            public double Activation(double value)
            {
                return Math.Tanh(value);
            }

            public double Deactivation(double value)
            {
                return 1.0 - Math.Pow(Math.Tanh(value), 2.0);
            }
        }

        private class ActivationLeakyRelu : IActivationMethods
        {
            public double Activation(double value)
            {
                return Math.Max(0.1 * value, value);
            }

            public double Deactivation(double value)
            {
                return value >= 0 ? 1 : 0.01;
            }
        }
        #endregion

        public class TrainSet
        {
            struct dataStruct
            {
                public double[] input;
                public double[] output;
            }

            public int inputSize;
            public int outputSize;
            private NetworkTools tools;

            private List<dataStruct> allData;

            public TrainSet(int INPUT_SIZE, int OUTPUT_SIZE)
            {
                this.inputSize = INPUT_SIZE;
                this.outputSize = OUTPUT_SIZE;
                tools = new NetworkTools();
                allData = new List<dataStruct>();
            }

            public void AddData(double[] dataIn, double[] expected)
            {
                if (dataIn.Length != inputSize || expected.Length != outputSize) return;
                allData.Add(new dataStruct { input = dataIn, output = expected });
            }

            public void ClearData()
            {
                allData = new List<dataStruct>();
            }

            // Fisher-Yates shuffle algorithm
            public void ShuffleData()
            {
                Random rand = new();
                int n = allData.Count;
                while (n > 1)
                {
                    n--;
                    int k = rand.Next(n + 1);
                    var value = allData[k];
                    allData[k] = allData[n];
                    allData[n] = value;
                }
            }

            public TrainSet ExtractBatch(int size)
            {
                if (size <= 0 || size > this.Size()) return this;
                var set = new TrainSet(inputSize, outputSize);
                var ids = tools.RandomValues(0, this.Size(), size);
                foreach (var temp in ids)
                {
                    set.AddData(GetInput(temp), GetOutput(temp));
                }

                return set;
            }

            public int Size()
            {
                return allData.Count;
            }

            public double[] GetInput(int index)
            {
                if (index >= 0 && index < Size())
                    return allData[index].input;
                return null;
            }

            public double[] GetOutput(int index)
            {
                if (index >= 0 && index < Size())
                    return allData[index].output;
                return null;
            }

            public int GetInputSize()
            {
                return inputSize;
            }

            public int GetOutputSize()
            {
                return outputSize;
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

            public void ApplyMutation(ref double[][][] weights, double probability)
            {
                for (int layer = 1; layer < weights.GetLength(0); layer++)
                {
                    for (int neuron = 0; neuron < weights[layer].Length; neuron++)
                    {
                        for (int single = 0; single < weights[layer][neuron].Length; single++)
                        {
                            double chance = RandomValue(0, 1);
                            if (probability >= chance)
                            {
                                double value = RandomValue(-1, 1);
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
                        double chance = RandomValue(0, 1);
                        if (probability >= chance)
                        {
                            double value = RandomValue(-1, 1);
                            bias[layer][neuron] = value;
                        }
                    }
                }
            }

            public double[] CreateRandomArray(int size, double lower_bound, double upper_bound)
            {
                if (size < 1) return null;
                var ar = new double[size];
                for (var i = 0; i < ar.Length; i++)
                {
                    ar[i] = RandomValue(lower_bound, upper_bound);
                }

                return ar;
            }

            public double[][] CreateRandomArray(int sizeX, int sizeY, double lower_bound, double upper_bound)
            {
                if (sizeX < 1 || sizeY < 1) return null;
                var ar = new double[sizeX][];
                for (var i = 0; i < sizeX; i++)
                {
                    ar[i] = CreateRandomArray(sizeY, lower_bound, upper_bound);
                }

                return ar;
            }

            public double RandomValue(double lower_bound, double upper_bound)
            {
                return rand.NextDouble() * (upper_bound - lower_bound) + lower_bound;
            }

            private bool Contais(int[] values, int value)
            {
                foreach (var i in values)
                {
                    if (i == value) return true;
                }

                return false;
            }

            public int[] RandomValues(int lowerBound, int upperBound, int amount)
            {
                lowerBound--;

                if (amount > (upperBound - lowerBound)) return null;

                var values = new int[amount];
                for (var n = 0; n < values.Length; n++)
                {
                    var i = rand.Next(lowerBound + 1, upperBound);
                    while (Contais(values, i))
                    {
                        i = rand.Next(lowerBound + 1, upperBound);
                    }

                    values[n] = i;
                }

                return values;
            }

            public static int IndexOfHighestValue(double[] values)
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

            public static int IndexOfHighestValue(double[] values, int from, int to)
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
