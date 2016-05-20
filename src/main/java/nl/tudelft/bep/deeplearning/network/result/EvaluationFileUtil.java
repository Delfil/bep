package nl.tudelft.bep.deeplearning.network.result;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.eval.Evaluation;

import nl.tudelft.bep.deeplearning.network.builder.FNNCBuilder;
import nl.tudelft.bep.deeplearning.network.data.Data;

public class EvaluationFileUtil {
	protected static final DecimalFormat SEED_FORMATER = new DecimalFormat(
			"S+0000000000000000000;S-0000000000000000000");
	protected static final DecimalFormat EPOCH_FORMATER = new DecimalFormat("E000");

	private static final String F = "/";
	private static final Object EVAL_EXTENTION = ".eval";

	public static String getEvalFileName(long seed, int epoch) {
		return new StringBuilder(SEED_FORMATER.format(seed)).append(EPOCH_FORMATER.format(epoch)).append(EVAL_EXTENTION)
				.toString();
	}

	public static String getEvalPathName(Data data, FNNCBuilder builder) {
		return new StringBuilder(builder.getPathName()).append(F).append(data.getTimeStamp()).append(F).toString();
	}

	/**
	 * Load a evaluation file from the disk.
	 * 
	 * @param seed
	 *            the initialization seed used to compute this evaluation
	 * @param epoch
	 *            the number of epochs used to compute this evaluation
	 * @param data
	 *            the data used to compute this evaluation
	 * @param builder
	 *            the network configuration used to compute this evaluation
	 * @return the requested evaluation file if it exists<br>
	 *         {@code null} if it doesn't exists
	 */
	public static Evaluation<Double> load(long seed, int epoch, Data data, FNNCBuilder builder) {
		File file = new File(getEvalPathName(data, builder) + getEvalFileName(seed, epoch));
		if (file.exists()) {
			try {
				FileInputStream fileIn = new FileInputStream(file);
				ObjectInputStream in = new ObjectInputStream(fileIn);
				@SuppressWarnings("unchecked")
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

	/**
	 * Load all saved evaluations which are generated with the same data,
	 * network configuration and epoch count.
	 * 
	 * @param epoch
	 *            the number of epochs used to compute these evaluations
	 * @param data
	 *            the data used to compute these evaluations
	 * @param builder
	 *            the network configuration used to compute these evaluations
	 * @return a list of all saved evaluations which satisfy the given
	 *         conditions
	 */
	public static List<Evaluation<Double>> load(int epoch, Data data, FNNCBuilder builder) {
		File folder = new File(getEvalPathName(data, builder));
		if (!folder.exists() || folder.isFile()) {
			return new ArrayList<>();
		}
		List<Evaluation<Double>> results = new ArrayList<>();
		String suffix = EPOCH_FORMATER.format(epoch) + EVAL_EXTENTION;
		try {
			for (File file : folder.listFiles()) {
				if (file.getName().endsWith(suffix)) {
					FileInputStream fileIn = new FileInputStream(file);
					ObjectInputStream in = new ObjectInputStream(fileIn);
					@SuppressWarnings("unchecked")
					Evaluation<Double> eval = (Evaluation<Double>) in.readObject();
					in.close();
					fileIn.close();
					results.add(eval);
				}
			}
		} catch (IOException | ClassNotFoundException e) {
			e.printStackTrace();
		}
		return results;
	}

	public static boolean evalExistst(long seed, int epoch, Data data, FNNCBuilder builder) {
		return new File(getEvalPathName(data, builder) + getEvalFileName(seed, epoch)).exists();
	}

	/**
	 * Save a evaluation to the disk.
	 * 
	 * @param eval
	 *            the evaluation to save
	 * @param seed
	 *            the initialization seed used to compute this evaluation
	 * @param epoch
	 *            the number of epochs used to compute this evaluation
	 * @param data
	 *            the data used to compute this evaluation
	 * @param builder
	 *            the network configuration used to compute this evaluation
	 * 
	 */
	public static void save(Evaluation<Double> eval, long seed, int epoch, Data data, FNNCBuilder builder) {
		String pathName = getEvalPathName(data, builder);
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
}
