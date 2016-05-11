package nl.tudelft.bep.deeplearning;

import java.io.IOException;
import java.text.DateFormat;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Random;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import nl.tudelft.bep.deeplearning.cnn.CNN;
import nl.tudelft.bep.deeplearning.datafetcher.DataPath;
import nl.tudelft.bep.deeplearning.datafetcher.MatrixDatasetIterator;

public class Tester {
	private static final Logger log = LoggerFactory.getLogger(Tester.class);
	private static final long SEEDER_SEED = 0;
	protected static final DecimalFormat SEED_FORMATER = new DecimalFormat("S0000000000000000000");
	protected static final DecimalFormat EPOCH_FORMATER = new DecimalFormat("E000");
	protected static final DateFormat TIME_FORMATER = new SimpleDateFormat("HH:mm:ss:SSS");
	protected static final DateFormat TIME_FORMATER_HM = new SimpleDateFormat("HH:mm");

	public static void main(String[] args) throws IOException, ClassNotFoundException {
		System.out.println(new DenseLayer.Builder().activation("relu").nOut(50).build().toString());

		NNConfigurationBuilder builder = CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(1, 1).stride(1, 1).nOut(1).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(50).build(),
				new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax")
						.build());

		Tester test = new Tester(builder, DataPath.readDataSet("100_Genes/100_Genes"));
		test.start(1, 5);
	}

	protected final Random seeder;
	protected final String networkPath;
	protected final NNConfigurationBuilder builder;
	protected final DataSetIterator trainIterator;
	protected final DataSetIterator testIterator;
	protected final DataPath data;

	public Tester(NNConfigurationBuilder builder, DataPath data) {
		this.seeder = new Random(SEEDER_SEED);
		this.data = data;
		this.builder = builder;
		this.networkPath = NetworkUtil.getPathName(builder);
		builder.save();
		this.trainIterator = new MatrixDatasetIterator(this.data, 0.0, this.data.getTrainPercentage());
		this.testIterator = new MatrixDatasetIterator(this.data, this.data.getTrainPercentage(), 1.0);
	}

	protected MultiLayerNetwork setupModel(long seed) {
		org.deeplearning4j.nn.conf.MultiLayerConfiguration.Builder listBuilder = builder.list();
		new ConvolutionLayerSetup(listBuilder, this.data.getWidth(), this.data.getHeight(), 1);

		MultiLayerConfiguration conf = listBuilder.build();
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}

	public void start(int iterations, int epochs) {
		long startTime = System.currentTimeMillis();
		for (int i = 1; i <= iterations; i++) {
			iterate(seeder.nextLong(), epochs);
			log.info("*** Completed iteration {}/{} ***", i, iterations);
			log.info("*** Time remainding: {}/{} ***",
					TIME_FORMATER.format(new Date(System.currentTimeMillis() - startTime)),
					TIME_FORMATER_HM.format(new Date((System.currentTimeMillis() - startTime) / i * iterations)));
		}
	}

	protected void iterate(long seed, int epochs) {
		// if (data.getFileName()) //TODO: return if already computed
		// return;

		MultiLayerNetwork model = setupModel(seed);

		Evaluation<Double> eval = null;
		log.info("Train model on seed {}", seed);
		for (int i = 1; i <= epochs; i++) {
			model.fit(trainIterator);
			log.info("*** Completed epoch {}/{} ***", i, epochs);
		}
		eval = evaluate(model);
		save(eval, seed, epochs);
		log.info(eval.stats());
	}

	private Evaluation<Double> evaluate(MultiLayerNetwork model) {
		log.info("Evaluate model....");
		Evaluation<Double> eval = new Evaluation<>(data.getNumOutcomes());
		this.testIterator.reset();
		while (this.testIterator.hasNext()) {
			org.nd4j.linalg.dataset.DataSet ds = this.testIterator.next();
			INDArray output = model.output(ds.getFeatureMatrix());
			eval.eval(ds.getLabels(), output);
		}
		return eval;
	}

	protected void save(Evaluation<Double> eval, long seed, int epoch) {
		String fileName = SEED_FORMATER.format(seed) + EPOCH_FORMATER.format(epoch);
		System.out.println(this.data.getPath());
		System.out.println(this.networkPath);
		System.out.println(fileName);
	}
}