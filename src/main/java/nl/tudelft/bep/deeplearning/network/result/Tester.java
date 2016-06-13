package nl.tudelft.bep.deeplearning.network.result;

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

import nl.tudelft.bep.deeplearning.network.builder.FNNCBuilder;
import nl.tudelft.bep.deeplearning.network.data.LoadedGeneExpressionDatabase;
import nl.tudelft.bep.deeplearning.network.data.GeneExpressionDatabase;
import nl.tudelft.bep.deeplearning.network.data.MatrixDatasetIterator;

public class Tester {
	protected static final Logger LOG = LoggerFactory.getLogger(Tester.class);
	protected static final long SEEDER_SEED = 0;
	protected static final DateFormat TIME_FORMATER = new SimpleDateFormat("HH:mm:ss:SSS");
	protected static final DateFormat TIME_FORMATER_HM = new SimpleDateFormat("HH:mm");

	private Random seeder;
	private final FNNCBuilder builder;
	private final DataSetIterator trainIterator;
	private final DataSetIterator testIterator;
	private final GeneExpressionDatabase data;

	public Tester(final FNNCBuilder builder, final GeneExpressionDatabase data) {
		this.data = data;
		this.builder = builder;

		this.trainIterator = new MatrixDatasetIterator(this.data, 0.0, this.data.getTrainPercentage());
		this.testIterator = new MatrixDatasetIterator(this.data, this.data.getTrainPercentage(), 1.0);
	}

	public Tester(final String networkFile, final String dataFile) {
		this(FNNCBuilder.load(networkFile), LoadedGeneExpressionDatabase.Loader.load(dataFile));
	}

	/**
	 * Setup a model with the given seed.
	 * 
	 * @param seed
	 *            the seed to set
	 * @return a configured and initialized model
	 */
	protected MultiLayerNetwork setupModel(final long seed) {
		this.builder.setSeed(seed);
		org.deeplearning4j.nn.conf.MultiLayerConfiguration.Builder listBuilder = this.builder.build();
		new ConvolutionLayerSetup(listBuilder, this.data.getWidth(), this.data.getHeight(), 1);

		MultiLayerConfiguration conf = listBuilder.build();
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}

	/**
	 * Start a {@link Tester}.
	 * 
	 * @param iterations
	 *            the number of iterations with different initialization seeds
	 *            to run
	 * @param epochs
	 *            the number of epochs to run each iteration
	 */
	public void start(final int iterations, final int epochs) {
		this.seeder = new Random(SEEDER_SEED);
		long startTime = System.currentTimeMillis();
		for (int i = 1; i <= iterations; i++) {
			if (this.iterate(this.seeder.nextLong(), epochs)) {
				LOG.info("*** Completed iteration {}/{} ***", i, iterations);
				LOG.info("*** Time remainding: {}/{} ***",
						TIME_FORMATER.format(new Date(System.currentTimeMillis() - startTime)),
						TIME_FORMATER_HM.format(new Date((System.currentTimeMillis() - startTime) / i * iterations)));
			} else {
				LOG.info("*** Skiped iteration {}/{} ***", i, iterations);
			}
		}
	}

	/**
	 * Start an iteration.
	 * 
	 * @param seed
	 *            the seed to run the iteration on
	 * @param epochs
	 *            the number of epochs to run this iteration
	 * @return {@code true} if the iteration successfully ran <br>
	 *         {@code false} if the iteration was already ran
	 */
	protected boolean iterate(final long seed, final int epochs) {
		if (EvaluationFileUtil.evalExistst(seed, epochs, this.data, this.builder)) {
			return false;
		}

		MultiLayerNetwork model = this.setupModel(seed);

		LOG.info("Train model on seed {}", seed);
		for (int i = 1; i <= epochs; i++) {
			model.fit(this.trainIterator);
			LOG.info("*** Completed epoch {}/{} ***", i, epochs);
		}
		Evaluation<Double> eval = this.evaluate(model);

		EvaluationFileUtil.save(eval, seed, epochs, this.data, this.builder);
		LOG.info(eval.stats());
		return true;
	}

	/**
	 * Computes a evaluation for the given model.
	 * 
	 * @param model
	 *            the model to evaluate
	 * @return a {@link Evaluation} of the given model
	 */
	protected Evaluation<Double> evaluate(final MultiLayerNetwork model) {
		LOG.info("Evaluate model....");
		Evaluation<Double> eval = new Evaluation<>(this.data.getNumOutcomes());
		this.testIterator.reset();
		while (this.testIterator.hasNext()) {
			org.nd4j.linalg.dataset.DataSet ds = this.testIterator.next();
			INDArray output = model.output(ds.getFeatureMatrix());
			eval.eval(ds.getLabels(), output);
		}
		return eval;
	}
}