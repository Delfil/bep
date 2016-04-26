package nl.tudelft.bep.deeplearning;

import java.io.DataOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class App {

    private static final long SEED = 39458L;
	private static Logger log = LoggerFactory.getLogger(App.class);
	
	public static void main(String[] s) throws IOException, InterruptedException {
		String name = "sampleData.txt";
		DataSetIterator data = getData(name, 100);
		int numInputs = 2 * 2, numOutput = 5, iterations = 1;
		MultiLayerNetwork model = getModel(numInputs, numOutput, iterations);
		train(model, data, 1);
//		saveNetwork(model, name);
	}

	private static void saveNetwork(MultiLayerNetwork model, String name) throws IOException {
		// Write the network parameters:
		try (DataOutputStream dos = new DataOutputStream(
				Files.newOutputStream(Paths.get(name + "/coefficients.bin")))) {
			Nd4j.write(model.params(), dos);
		}

		// Write the network configuration:
		FileUtils.write(new File(name + "/conf.json"), model.getLayerWiseConfigurations().toJson());
	}

	private static void train(MultiLayerNetwork model, DataSetIterator data, int nEpochs) {
		log.info("Train model....");
		model.setListeners(new ScoreIterationListener(1));
		for (int i = 0; i < nEpochs; i++) {
			model.fit(data);
			log.info("*** Completed epoch {} ***", i);

			log.info("Evaluate model....");
			int outputNum;
			Evaluation eval = new Evaluation(outputNum = 3);
			while (data.hasNext()) {
				DataSet ds = data.next();
				INDArray output = model.output(ds.getFeatureMatrix());
				eval.eval(ds.getLabels(), output);
			}
			log.info(eval.stats());
			data.reset();
		}
	}

	private static MultiLayerNetwork getModel(int numInput, int numOutput, int iterations) {
		log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(SEED)
                .iterations(iterations)
                .regularization(true).l2(0.0005)
                .learningRate(0.01)//.biasLearningRate(0.02)
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list(6)
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(1)
                        .stride(1, 1)
                        .nOut(20)
                        .activation("identity")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .nIn(1)
                        .stride(1, 1)
                        .nOut(50)
                        .activation("identity")
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder()
                		.activation("relu")
                        .nOut(50).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(5)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);
        new ConvolutionLayerSetup(builder,28,28,1);

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
		return model;
	}

	private static DataSetIterator getData(String fileName, int batchSize) throws IOException, InterruptedException {
		return new MatrixDatasetIterator(batchSize, new MatrixDataFetcher("/home/sam/bep/deeplearning/src/main/matlab/100data", false, SEED));
	}
}
