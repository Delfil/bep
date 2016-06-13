package nl.tudelft.bep.deeplearning.network.result;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import nl.tudelft.bep.deeplearning.network.builder.CNN;
import nl.tudelft.bep.deeplearning.network.builder.FNNCBuilder;
import nl.tudelft.bep.deeplearning.network.builder.NNCBuilder;

public class TesterTest {

	private String networkFile;
	private String dataFile = "test_data/correct_1";
	private String networkPath;

	@Before
	public void before() throws IOException {
		FNNCBuilder builder = this.getTestBuilder().finish();
		this.networkFile = builder.getFileName();
		this.networkPath = builder.getPathName();
		if (new File(this.networkPath).exists()) {
			FileUtils.cleanDirectory(new File(this.networkPath));
		}
	}

	@After
	public void after() throws IOException {
		new File(this.networkFile).delete();
		FileUtils.cleanDirectory(new File(this.networkPath));
		new File(this.networkPath).delete();
	}

	@Test
	public void test() {
		Tester tester = new Tester(this.networkFile, this.dataFile);
		tester.start(1, 1);
		assertEquals(1, new File(this.networkPath).listFiles()[0].listFiles().length);
		tester.start(2, 1);
		assertEquals(2, new File(this.networkPath).listFiles()[0].listFiles().length);
		tester.start(2, 2);
		assertEquals(4, new File(this.networkPath).listFiles()[0].listFiles().length);
	}

	protected NNCBuilder getTestBuilder() {
		NNCBuilder builder = CNN.buildExampleCNN(
				new ConvolutionLayer.Builder().kernelSize(1, 1).stride(1, 1).nOut(2).build(),
				new OutputLayer.Builder().nOut(2).build());
		builder.backprop(true);
		builder.pretrain(false);
		return builder;
	}
}
