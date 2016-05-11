package nl.tudelft.bep.deeplearning;

import java.io.File;
import java.util.stream.Collectors;

import org.deeplearning4j.nn.conf.layers.Layer;

public class NetworkUtil {

	protected final static String NETWORK_FOLDER = "networks";
	protected final static String MULTI_LAYER_NETWORK = "MLN_";
	protected final static String F = "/";
	protected static final String FILE_EXTENTION = ".NNConf.json";

	public static String getPathName(NNConfigurationBuilder builder) {
		String fileName = new StringBuilder(NETWORK_FOLDER).append(F).append(getShortDescription(builder)).append(F)
				.toString();
		File file = new File(fileName);
		file.mkdirs();
		File[] files = file.listFiles();
		int max = -1;
		for (File f : files) {
			String fn = f.getName();
			if (fn.endsWith(FILE_EXTENTION) && fn.startsWith(MULTI_LAYER_NETWORK)) {
				if (NNConfigurationBuilder.load(f.getAbsolutePath()).equals(builder)) {
					String string = f.getAbsolutePath();
					return string.substring(0, string.length() - FILE_EXTENTION.length());
				} else {
					max = Math.max(max,
							Integer.parseInt(fn.substring(MULTI_LAYER_NETWORK.length(), FILE_EXTENTION.length())));
				}
			}
		}
		fileName = new StringBuilder(fileName).append(F).append(MULTI_LAYER_NETWORK).append(Integer.toString(++max))
				.toString();
		return fileName;
	}

	public static String getNetworkFileName(NNConfigurationBuilder builder) {
		return getPathName(builder) + FILE_EXTENTION;
	}

	public static String getShortDescription(NNConfigurationBuilder builder) {
		return MULTI_LAYER_NETWORK + builder.getLayers().stream().map(Layer::getClass).map(Class::getName)
				.map(name -> name.split("\\.")).map(array -> array[array.length - 1])
				.map(name -> name.split("Layer")[0]).collect(Collectors.joining("|"));
	}

}
