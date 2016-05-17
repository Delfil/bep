package nl.tudelft.bep.deeplearning.test;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Random;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import nl.tudelft.bep.deeplearning.data.Data;
import nl.tudelft.bep.deeplearning.data.MatrixDatasetIterator;
import nl.tudelft.bep.deeplearning.network.FinishedNNCBuilder;

public class Tester {
	private static final Logger log = LoggerFactory.getLogger(Tester.class);
	protected static final long SEEDER_SEED = 0;
	protected static final DateFormat TIME_FORMATER = new SimpleDateFormat("HH:mm:ss:SSS");
	protected static final DateFormat TIME_FORMATER_HM = new SimpleDateFormat("HH:mm");

	protected final Random seeder;
	protected final FinishedNNCBuilder builder;
	protected final DataSetIterator trainIterator;
	protected final DataSetIterator testIterator;
	protected final Data data;

	public Tester(FinishedNNCBuilder builder, Data data) {
		this.seeder = new Random(SEEDER_SEED);
		this.data = data;
		this.builder = builder;

		this.trainIterator = new MatrixDatasetIterator(this.data, 0.0, this.data.getTrainPercentage());
		this.testIterator = new MatrixDatasetIterator(this.data, this.data.getTrainPercentage(), 1.0);
	}

	public Tester(String networkFile, String dataFile) {
		this(FinishedNNCBuilder.load(networkFile), Data.readDataSet(dataFile));
	}

	protected MultiLayerNetwork setupModel(long seed) {
		org.deeplearning4j.nn.conf.MultiLayerConfiguration.Builder listBuilder = builder.build();
		new ConvolutionLayerSetup(listBuilder, this.data.getWidth(), this.data.getHeight(), 1);

		MultiLayerConfiguration conf = listBuilder.build();
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}

	public void start(int iterations, int epochs) {
		long startTime = System.currentTimeMillis();
		for (int i = 1; i <= iterations; i++) {
			if (iterate(seeder.nextLong(), epochs)) {
				log.info("*** Completed iteration {}/{} ***", i, iterations);
				log.info("*** Time remainding: {}/{} ***",
						TIME_FORMATER.format(new Date(System.currentTimeMillis() - startTime)),
						TIME_FORMATER_HM.format(new Date((System.currentTimeMillis() - startTime) / i * iterations)));
			} else {
				log.info("*** Skiped iteration {}/{} ***", i, iterations);
			}
		}
	}

	protected boolean iterate(long seed, int epochs) {
		if (EvaluationFileUtil.evalExistst(seed, epochs, this.data, this.builder)) {
			return false;
		}

		MultiLayerNetwork model = setupModel(seed);

		log.info("Train model on seed {}", seed);
		for (int i = 1; i <= epochs; i++) {
			model.fit(trainIterator);
			log.info("*** Completed epoch {}/{} ***", i, epochs);
		}
		Evaluation<Double> eval = evaluate(model);

		EvaluationFileUtil.save(eval, seed, epochs, this.data, this.builder);
//		log.info(eval.stats());
		return true;
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
}