package nl.tudelft.bep.deeplearning.network.result;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import nl.tudelft.bep.deeplearning.network.builder.CNN;
import nl.tudelft.bep.deeplearning.network.builder.FNNCBuilder;
import nl.tudelft.bep.deeplearning.network.data.Data;
import nl.tudelft.bep.deeplearning.network.result.csv.CSVFiller;
import nl.tudelft.bep.deeplearning.network.result.csv.ComputeAverageAccuracyFiller;

public class ResultUtilTest {

	private List<String> toRemove = new ArrayList<>();
	private FNNCBuilder builder;
	private Data data;

	@Before
	public void before() {
		this.data = Data.readDataSet("test_data/correct_1");
		this.builder = CNN.buildExampleCNN(new ConvolutionLayer.Builder().kernelSize(1, 1).stride(1, 1).nOut(2).build(),
				new OutputLayer.Builder().nOut(2).build()).finish();
		this.toRemove.add(this.builder.getPathName());
		this.toRemove.add(this.builder.getFileName());
		this.toRemove.add(EvaluationFileUtil.getEvalPathName(this.data, this.builder));
	}

	@Test
	public void csvGenTest() {
		CountFiller countFiller = new CountFiller();
		Set<String> networks = new HashSet<>(ResultUtil.getNetworkList());
		Set<String> dataSets = new HashSet<>(ResultUtil.getDataList());
		int epochs = 1;
		ResultUtil.generateCSV(epochs, countFiller, "toRemove");
		this.toRemove.add("toRemove.csv");
		assertEquals(networks.size() * dataSets.size(), countFiller.getCount());
		assertEquals(networks, countFiller.getNetworks());
		assertEquals(dataSets, countFiller.getDataSets());
		assertEquals(1, countFiller.getEpochs().size());
		assertEquals(epochs, countFiller.getEpochs().iterator().next().intValue());
	}

	@Test
	public void computeAverageAccuracyFillerTest() {
		ComputeAverageAccuracyFiller caaf = new ComputeAverageAccuracyFiller(1);
		assertEquals("0.5", caaf.fill(this.builder.getFileName(), this.data.getName(), 1));
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

	class CountFiller implements CSVFiller {
		private Set<String> networks = new HashSet<>();
		private Set<String> dataSets = new HashSet<>();
		private Set<Integer> epochs = new HashSet<>();
		private int count = 0;

		@Override
		public String fill(final String network, final String data, final int epoch) {
			this.networks.add(network);
			this.dataSets.add(data);
			this.epochs.add(epoch);
			this.count++;
			return "";
		}

		public Set<String> getNetworks() {
			return this.networks;
		}

		public Set<String> getDataSets() {
			return this.dataSets;
		}

		public Set<Integer> getEpochs() {
			return this.epochs;
		}

		public int getCount() {
			return this.count;
		}
	}
}
