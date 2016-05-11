package nl.tudelft.bep.deeplearning;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.Layer;

import com.fasterxml.jackson.databind.ObjectMapper;

public class NNConfigurationBuilder extends NeuralNetConfiguration.Builder {

	protected boolean backprop = false;
	protected boolean pretrain = false;
	protected List<Layer> layers = new ArrayList<>();

	public void add(Layer layer) {
		layers.add(layer);
	}

	@Override
	@Deprecated
	public ListBuilder list(int size) {
		return null;
	}

	@Override
	@Deprecated
	public NeuralNetConfiguration build() {
		return super.build();
	}

	public NNConfigurationBuilder backprop(boolean backprop) {
		this.backprop = backprop;
		return this;
	}

	public NNConfigurationBuilder pretrain(boolean pretrain) {
		this.pretrain = pretrain;
		return this;
	}

	public org.deeplearning4j.nn.conf.MultiLayerConfiguration.Builder list() {
		ListBuilder lb = super.list(this.layers.size());
		for (int i = 0; i < layers.size(); i++) {
			lb.layer(i, layers.get(i));
		}
		lb.setBackprop(this.backprop);
		lb.setPretrain(this.pretrain);
		return lb;
	}

	public List<Layer> getLayers() {
		return layers;
	}

	public void save() {
		ObjectMapper mapper = new ObjectMapper();
		try {
			Files.write(Paths.get(NetworkUtil.getNetworkFileName(this)),
					mapper.writeValueAsString(this).replaceAll(",", ",\n").getBytes(StandardCharsets.UTF_8));
		} catch (IOException e1) {
			e1.printStackTrace();
		}
	}

	public static NNConfigurationBuilder load(String fileName) {
		ObjectMapper mapper = new ObjectMapper();
		try {
			return mapper.readValue(new File(fileName), NNConfigurationBuilder.class);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
	}
}
