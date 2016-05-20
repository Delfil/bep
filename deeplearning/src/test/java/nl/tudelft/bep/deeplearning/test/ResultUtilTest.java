package nl.tudelft.bep.deeplearning.test;

import static org.junit.Assert.*;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.DataSets;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import nl.tudelft.bep.deeplearning.data.Data;
import nl.tudelft.bep.deeplearning.network.CNN;
import nl.tudelft.bep.deeplearning.network.FNNCBuilder;

public class ResultUtilTest {

	private static final int EPOCH = 0;
	private static final long SEED = 0;
	private List<String> toRemove = new ArrayList<>();
	private FNNCBuilder builder;
	private Data data;

	@Before
	public void before() {
		data = Data.readDataSet("test_data/correct_1");
		builder = CNN.BuildExampleCNN(new ConvolutionLayer.Builder().kernelSize(1, 1).stride(1, 1).nOut(2).build(),
				new OutputLayer.Builder().nOut(2).build()).finish();
		toRemove.add(builder.getPathName());
		toRemove.add(builder.getFileName());
		toRemove.add(EvaluationFileUtil.getEvalPathName(data, builder));
	}

	@Test
	public void csvGenTest() {
		CountFiller countFiller = new CountFiller();
		Set<String> networks = new HashSet<>(ResultUtil.getNetworkList());
		Set<String> dataSets = new HashSet<>(ResultUtil.getDataList());
		int epochs = 1;
		ResultUtil.generateCSV(epochs, countFiller, "toRemove");
		toRemove.add("toRemove.csv");
		assertEquals(networks.size() * dataSets.size(), countFiller.getCount());
		assertEquals(networks, countFiller.getNetworks());
		assertEquals(dataSets, countFiller.getDataSets());
		assertEquals(1, countFiller.getEpochs().size());
		assertEquals(epochs, countFiller.getEpochs().iterator().next().intValue());
	}

	@Test
	public void ComputeAverageAccuracyFillerTest() {
		ComputeAverageAccuracyFiller caaf = new ComputeAverageAccuracyFiller(1);
		assertEquals("0.5", caaf.fill(this.builder.getFileName(), this.data.getName(), 1));
	}

	@After
	public void after() {
		toRemove.forEach(name -> {
			File file = new File(name);
			if (!file.exists())
				return;
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
		public String fill(String network, String data, int epoch) {
			networks.add(network);
			dataSets.add(data);
			epochs.add(epoch);
			count++;
			return "";
		}

		public Set<String> getNetworks() {
			return networks;
		}

		public Set<String> getDataSets() {
			return dataSets;
		}

		public Set<Integer> getEpochs() {
			return epochs;
		}

		public int getCount() {
			return count;
		}
	}
}
