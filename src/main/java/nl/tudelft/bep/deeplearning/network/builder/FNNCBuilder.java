package nl.tudelft.bep.deeplearning.network.builder;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.stream.Collectors;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.Layer;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

public class FNNCBuilder {
	public final static String NETWORK_FOLDER = "networks";
	public final static String MULTI_LAYER_NETWORK = "MLN_";
	protected final static String F = "/";
	public static final String NETWORK_SUFFIX = ".NNConf.json";
	public static final String DESCRIPTION_SUFFIX = ".txt";

	protected final String fileName;
	protected final String pathName;
	protected final NNCBuilder builder;
	private String description;

	public FNNCBuilder(NNCBuilder nncBuilder, String description) {
		this.builder = nncBuilder.clone();
		this.pathName = this.computePathName();
		this.fileName = this.pathName + NETWORK_SUFFIX;
		this.description = description;
		this.save();
	}

	public FNNCBuilder(NNCBuilder nncBuilder) {
		this.builder = nncBuilder.clone();
		this.pathName = this.computePathName();
		this.fileName = this.pathName + NETWORK_SUFFIX;
		this.description = getDescription(fileName);
		this.save();
	}

	/**
	 * Compute the path name that should be used to save or load this
	 * {@link FNNCBuilder}.
	 * 
	 * @return the path name corresponding to this {@link FNNCBuilder} instance
	 */
	protected String computePathName() {
		String fileName = new StringBuilder(NETWORK_FOLDER).append(F).append(this.getShortDescription()).append(F)
				.toString();
		File file = new File(fileName);
		file.mkdirs();
		File[] files = file.listFiles();
		int max = -1;
		String thisBuilderString = toJSON(this.builder);
		for (File f : files) {
			String fn = f.getName();
			if (fn.endsWith(NETWORK_SUFFIX) && fn.startsWith(MULTI_LAYER_NETWORK)) {
				if (FNNCBuilder.toJSON(FNNCBuilder.loadBuilder(f.getAbsolutePath())).equals(thisBuilderString)) {
					String string = f.getAbsolutePath();
					return string.substring(0, string.length() - NETWORK_SUFFIX.length());
				} else {
					max = Math.max(max, Integer.parseInt(
							fn.substring(MULTI_LAYER_NETWORK.length(), fn.length() - NETWORK_SUFFIX.length())));
				}
			}
		}
		return new StringBuilder(fileName).append(MULTI_LAYER_NETWORK).append(Integer.toString(++max)).toString();
	}

	/**
	 * Converts the given {@link NNCBuilder} instance to a JSON {@link String}.
	 * 
	 * @param loadBuilder
	 *            the {@link NNCBuilder} instance to convert
	 * @return a {@link String}
	 */
	protected static String toJSON(NNCBuilder loadBuilder) {
		ObjectMapper mapper = new ObjectMapper();
		try {
			return mapper.writeValueAsString(loadBuilder);
		} catch (JsonProcessingException e) {
			e.printStackTrace();
		}
		return null;
	}

	/**
	 * Save this network configuration to the disk.
	 */
	protected void save() {
		ObjectMapper mapper = new ObjectMapper();
		mapper.enable(SerializationFeature.INDENT_OUTPUT);
		try {
			Files.write(Paths.get(this.fileName),
					mapper.writeValueAsString(this.builder).replaceAll(",", ",\n").getBytes(StandardCharsets.UTF_8));

			PrintWriter pw = new PrintWriter(new File(this.pathName + DESCRIPTION_SUFFIX));
			pw.println(this.description);
			pw.close();
		} catch (IOException e1) {
			e1.printStackTrace();
		}
	}

	/**
	 * Initialize a {@link FNNCBuilder} instance from a saved {@link NNCBuilder}
	 * .
	 * 
	 * @param fileName
	 *            the fileName of the instance to load
	 * @return a {@link FNNCBuilder} instance
	 */
	public static FNNCBuilder load(String fileName) {
		return new FNNCBuilder(loadBuilder(fileName));
	}

	/**
	 * Load a saved {@link NNCBuilder} instance.
	 * 
	 * @param fileName
	 *            the fileName of the instance to load
	 * @return the saved {@link NNCBuilder} instance
	 */
	protected static NNCBuilder loadBuilder(String fileName) {
		ObjectMapper mapper = new ObjectMapper();
		try {
			return mapper.readValue(new File(fileName), NNCBuilder.class);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
	}

	/**
	 * Generates a short description of the network configuration as a sum of
	 * layer names within the network configuration.
	 * 
	 * @return a {@link String} with a short description of the instance
	 */
	public String getShortDescription() {
		return MULTI_LAYER_NETWORK + this.builder.getLayers().stream().map(Layer::getClass).map(Class::getName)
				.map(name -> name.split("\\.")).map(array -> array[array.length - 1])
				.map(name -> name.split("Layer")[0]).collect(Collectors.joining("_"));
	}

	/**
	 * Use the set configurations to build a
	 * {@link org.deeplearning4j.nn.conf.MultiLayerConfiguration.Builder }.
	 * 
	 * @return a configured
	 *         {@link org.deeplearning4j.nn.conf.MultiLayerConfiguration.Builder }
	 *         instance
	 */
	public org.deeplearning4j.nn.conf.MultiLayerConfiguration.Builder build() {
		int layersC = this.builder.getLayers().size();
		ListBuilder lb = this.builder.list(layersC);
		for (int i = 0; i < layersC; i++) {
			lb.layer(i, this.builder.getLayers().get(i));
		}
		lb.setBackprop(this.builder.backprop);
		lb.setPretrain(this.builder.pretrain);
		return lb;
	}

	public void setSeed(long seed) {
		this.builder.seed(seed);
	}

	public String getPathName() {
		return this.pathName;
	}

	public String getFileName() {
		return this.fileName;
	}

	public static String getDescription(String builderFileName) {
		try {
			BufferedReader bufferedReader = new BufferedReader(
					new FileReader(builderFileName.substring(0, builderFileName.length() - NETWORK_SUFFIX.length())
							+ DESCRIPTION_SUFFIX));
			String r = bufferedReader.readLine();
			bufferedReader.close();
			return r;
		} catch (IOException e) {
			return "";
		}
	}
}
