package nl.tudelft.bep.deeplearning.network;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.stream.Collectors;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.Layer;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

public class FinishedNNCBuilder {
	public final static String NETWORK_FOLDER = "networks";
	protected final static String MULTI_LAYER_NETWORK = "MLN_";
	protected final static String F = "/";
	public static final String NETWORK_SUFFIX = ".NNConf.json";

	protected final String fileName;
	protected final String pathName;
	protected final NNCBuilder builder;

	public FinishedNNCBuilder(NNCBuilder nncBuilder) {
		this.builder = nncBuilder.clone();
		this.pathName = this.computePathName();
		this.fileName = this.pathName + NETWORK_SUFFIX;
		this.save();
	}

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
				if (FinishedNNCBuilder.toJSON(FinishedNNCBuilder.loadBuilder(f.getAbsolutePath()))
						.equals(thisBuilderString)) {
					String string = f.getAbsolutePath();
					return string.substring(0, string.length() - NETWORK_SUFFIX.length());
				} else {
					max = Math.max(max, Integer.parseInt(
							fn.substring(MULTI_LAYER_NETWORK.length(), fn.length() - NETWORK_SUFFIX.length())));
				}
			}
		}
		return new StringBuilder(fileName).append(F).append(MULTI_LAYER_NETWORK).append(Integer.toString(++max))
				.toString();
	}

	private static String toJSON(NNCBuilder loadBuilder) {
		ObjectMapper mapper = new ObjectMapper();
		try {
			return mapper.writeValueAsString(loadBuilder);
		} catch (JsonProcessingException e) {
			e.printStackTrace();
		}
		return null;
	}

	protected void save() {
		ObjectMapper mapper = new ObjectMapper();
		mapper.enable(SerializationFeature.INDENT_OUTPUT);
		try {
			Files.write(Paths.get(this.fileName),
					mapper.writeValueAsString(this.builder).replaceAll(",", ",\n").getBytes(StandardCharsets.UTF_8));
		} catch (IOException e1) {
			e1.printStackTrace();
		}
	}

	public static FinishedNNCBuilder load(String fileName) {
		return new FinishedNNCBuilder(loadBuilder(fileName));
	}

	protected static NNCBuilder loadBuilder(String fileName) {
		ObjectMapper mapper = new ObjectMapper();
		try {
			return mapper.readValue(new File(fileName), NNCBuilder.class);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
	}

	public String getShortDescription() {
		return MULTI_LAYER_NETWORK + this.builder.getLayers().stream().map(Layer::getClass).map(Class::getName)
				.map(name -> name.split("\\.")).map(array -> array[array.length - 1])
				.map(name -> name.split("Layer")[0]).collect(Collectors.joining("_"));
	}

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

	public void seed(long seed) {
		this.builder.seed(seed);
	}

	public String getPathName() {
		return this.pathName;
	}

	public String getFileName() {
		return this.fileName;
	}
}
