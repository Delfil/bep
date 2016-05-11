package nl.tudelft.bep.deeplearning;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.text.DateFormat;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Random;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
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
	protected static final DecimalFormat SEED_FORMATER = new DecimalFormat(
			"S+0000000000000000000;S-0000000000000000000");
	protected static final DecimalFormat EPOCH_FORMATER = new DecimalFormat("E000");
	protected static final DateFormat TIME_FORMATER = new SimpleDateFormat("HH:mm:ss:SSS");
	protected static final DateFormat TIME_FORMATER_HM = new SimpleDateFormat("HH:mm");
	private static final String F = "/";
	private static final Object EVAL_EXTENTION = ".eval";

	public static void main(String[] args) throws IOException, ClassNotFoundException {

		NNConfigurationBuilder builder = CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(5, 5).stride(2, 2).nOut(10).activation("identity").build(),
				new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build(),
				// new DenseLayer.Builder().activation("relu").nOut(50).build(),
				new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax")
						.build());

		Tester test = new Tester(builder, DataPath.readDataSet("100_Genes/100_Genes"));
		test.start(20, 1);
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
		if (evalExistst(seed, epochs)) {
			return false;
		}

		MultiLayerNetwork model = setupModel(seed);

		log.info("Train model on seed {}", seed);
		for (int i = 1; i <= epochs; i++) {
			model.fit(trainIterator);
			log.info("*** Completed epoch {}/{} ***", i, epochs);
		}
		Evaluation<Double> eval = evaluate(model);
		
		save(eval, seed, epochs);
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

	protected Evaluation<Double> load(long seed, int epoch) {
		File file = new File(getEvalPathName() + getEvalFileName(seed, epoch));
		if (file.exists()) {
			try {
				FileInputStream fileIn = new FileInputStream(file);
				ObjectInputStream in = new ObjectInputStream(fileIn);
				Evaluation<Double> eval = (Evaluation<Double>) in.readObject();
				in.close();
				fileIn.close();
				return eval;
			} catch (IOException | ClassNotFoundException e) {
				e.printStackTrace();
			}
			return null;
		} else {
			return null;
		}
	}

	private boolean evalExistst(long seed, int epoch) {
		return new File(getEvalPathName() + getEvalFileName(seed, epoch)).exists();
	}

	protected void save(Evaluation<Double> eval, long seed, int epoch) {
		String pathName = getEvalPathName();
		File file = new File(pathName);
		if (!file.exists()) {
			file.mkdirs();
		}
		try {
			FileOutputStream fileOut = new FileOutputStream(new File(pathName + getEvalFileName(seed, epoch)));
			ObjectOutputStream oos = new ObjectOutputStream(fileOut);
			oos.writeObject(eval);
			oos.close();
			fileOut.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private String getEvalFileName(long seed, int epoch) {
		return new StringBuilder(SEED_FORMATER.format(seed)).append(EPOCH_FORMATER.format(epoch)).append(EVAL_EXTENTION)
				.toString();
	}

	private String getEvalPathName() {
		return new StringBuilder(this.networkPath).append(F).append(this.data.getTimeStamp()).append(F).toString();
	}
}