package nl.tudelft.bep.deeplearning.mlnn;

import java.io.IOException;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
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

public class MLNN {

	private static final Logger log = LoggerFactory.getLogger(MLNN.class);

	public static void main(String[] s) throws IOException {
		int batchSize = 100;
		int nEpochs = 10;
		int seed = 123;

		log.info("Load data....");

		String fileName = "/sample_dataAVGtest";
		MatrixDataFetcher fetcher = new MatrixDataFetcher(fileName, seed, 0.0, 0.75);
		DataSetIterator mnistTrain = new MatrixDatasetIterator(batchSize, fetcher);
		DataSetIterator mnistTest = new MatrixDatasetIterator(batchSize,
				new MatrixDataFetcher(fileName, seed, 0.75, 1.0));

		int width = fetcher.getWidth();
		int height = fetcher.getHeight();
		int outputNum = fetcher.getOutputNum();

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().iterations(1).learningRate(0.01)
				.regularization(true).l2(1e-4).list(3)
				.layer(0,
						new DenseLayer.Builder().nIn(width * height).nOut(50).activation("tanh")
								.weightInit(WeightInit.XAVIER).build())
				.layer(1,
						new DenseLayer.Builder().nIn(50).nOut(25).activation("tanh").weightInit(WeightInit.XAVIER)
								.build())
				.layer(2,
						new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
								.weightInit(WeightInit.XAVIER).activation("softmax").nIn(25).nOut(outputNum).build())
				.backprop(true).pretrain(false).build();

		// run the model
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		model.setListeners(new ScoreIterationListener(100));

		log.info("Train model....");
		model.setListeners(new ScoreIterationListener(1));
		for (int i = 0; i < nEpochs; i++) {
			model.fit(mnistTrain);
			log.info("*** Completed epoch {} ***", i);

			log.info("Evaluate model....");
			@SuppressWarnings("rawtypes")
			Evaluation eval = new Evaluation(outputNum);
			while (mnistTest.hasNext()) {
				DataSet ds = mnistTest.next();
				INDArray output = model.output(ds.getFeatureMatrix());
				eval.eval(ds.getLabels(), output);
			}
			log.info(eval.stats());
			mnistTest.reset();
		}
		log.info("****************Example finished********************");
	}
}
