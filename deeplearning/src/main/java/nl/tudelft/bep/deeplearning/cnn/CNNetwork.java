package nl.tudelft.bep.deeplearning.cnn;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import nl.tudelft.bep.deeplearning.datafetcher.MatrixDataFetcher;
import nl.tudelft.bep.deeplearning.datafetcher.MatrixDatasetIterator;

public class CNNetwork {

	private static final Logger log = LoggerFactory.getLogger(CNNetwork.class);

	public static void main(String[] args) throws Exception {
		int batchSize = 32;
		int nEpochs = 10;
		int trainingSeed = 666;
		int networkSeed = 333;


		log.info("Load data....");

		String fileName = "/100_Genes";
		MatrixDataFetcher fetcher = new MatrixDataFetcher(fileName, trainingSeed, 0.0, 0.75);
		DataSetIterator mnistTrain = new MatrixDatasetIterator(batchSize, fetcher);
		DataSetIterator mnistTest = new MatrixDatasetIterator(batchSize,
				new MatrixDataFetcher(fileName, trainingSeed, 0.75, 1.0));

		int width = fetcher.getWidth();
		int height = fetcher.getHeight();
		int outputNum = fetcher.getOutputNum();
		
		log.info("Build model....");
		MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().seed(networkSeed).iterations(1)
				.regularization(true).l2(0.0005).learningRate(0.01).weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.NESTEROVS)
				.momentum(0.9).list(3)
				.layer(0, new ConvolutionLayer.Builder(1, 1)
						.stride(1, 1)
						.nOut(1)
						.activation("identity").build())
//				.layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
//						.kernelSize(2, 2)
//						.stride(2, 2)
//						.build())
				.layer(1, new DenseLayer.Builder()
						.activation("relu")
						.nOut(50)
						.build())
				.layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.nOut(outputNum)
						.activation("softmax").build())
				.backprop(true).pretrain(false);

		
//		Builder builder = Test.main(null);
		
		new ConvolutionLayerSetup(builder, width, height, 1);

		MultiLayerConfiguration conf = builder.build();
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		
		log.info("Train model....");
		model.setListeners(new ScoreIterationListener(1));
		for (int i = 0; i < nEpochs; i++) {
			model.fit(mnistTrain);
			log.info("*** Completed epoch {} ***", i);

			log.info("Evaluate model....");
			Evaluation<Double> eval = new Evaluation<>(outputNum);
			while (mnistTest.hasNext()) {
				DataSet ds = mnistTest.next();
				INDArray output = model.output(ds.getFeatureMatrix());
				eval.eval(ds.getLabels(), output);
			}
			log.info(eval.stats());
			mnistTest.reset();
		}
		log.info("**************** Finished ********************");
	}
}
