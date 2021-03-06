package nl.tudelft.bep.deeplearning.network.result.lister;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import nl.tudelft.bep.deeplearning.network.builder.CNN;
import nl.tudelft.bep.deeplearning.network.builder.FNNCBuilder;
import nl.tudelft.bep.deeplearning.network.data.LoadedGeneExpressionDatabase;
import nl.tudelft.bep.deeplearning.network.data.GeneExpressionDatabase;
import nl.tudelft.bep.deeplearning.network.result.EvaluationFileUtil;
import nl.tudelft.bep.deeplearning.network.result.ResultUtil;

public class ListerTest {

	private List<String> toRemove = new ArrayList<>();
	private FNNCBuilder builder;
	private GeneExpressionDatabase data;

	@Before
	public void before() {
		this.data = LoadedGeneExpressionDatabase.Loader.load("test_data/correct_1");
		this.builder = CNN.buildExampleCNN(new ConvolutionLayer.Builder().kernelSize(1, 1).stride(1, 1).nOut(2).build(),
				new OutputLayer.Builder().nOut(2).build()).finish();
		this.toRemove.add(this.builder.getPathName());
		this.toRemove.add(this.builder.getFileName());
		this.toRemove.add(EvaluationFileUtil.getEvalPathName(this.data, this.builder));
	}

	@Test
	public void listTest() {
		for (int i = 0; i < 2; i++) {
			Evaluation<Double> ev = new Evaluation<>(1);

			ev.eval(1, i);

			EvaluationFileUtil.save(ev, i, 1, this.data, this.builder);
		}
		String string = new ListAccuracy().list(1, this.data.getName(), this.builder.getFileName());
		
		assertTrue(string.contains("0.0"));
		assertTrue(string.contains("1.0"));
	}

	@Test
	public void fileListTest() {
		for (int i = 0; i < 2; i++) {
			Evaluation<Double> ev = new Evaluation<>(1);

			ev.eval(1, i);

			EvaluationFileUtil.save(ev, i, 1, this.data, this.builder);
		}
		List<String> networks = new ArrayList<>();
		networks.add(this.builder.getFileName());
		List<String> dataSets = new ArrayList<>();
		dataSets.add(this.data.getName());

		this.toRemove.add("test");
		ResultUtil.generateLists(1, new ListAccuracy(), "test", networks, dataSets);
		File file = new File("test/test_data/correct_1/networks/MLN_Convolution_Output/MLN_0.NNConf.json.csv");
		assertTrue(file.exists());
		try {
			String string = new String(Files.readAllBytes(Paths.get(file.getAbsolutePath().replace("\\", "/"))));
			assertTrue(string.contains("\nnetworks/MLN_Convolution_Output/MLN_0.NNConf.json\ntest_data/correct_1"));
			assertTrue(string.contains("\n0.0"));
			assertTrue(string.contains("\n1.0"));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@After
	public void after() {
		this.toRemove.forEach(name -> {
			File file = new File(name);
			if (!file.exists()) {
				return;
			}
			if (file.isFile()) {
				file.delete();
			} else {
				try {
					FileUtils.cleanDirectory(file);
				} catch (Exception e) {
					e.printStackTrace();
				}
				file.delete();
			}
		});
	}
}
