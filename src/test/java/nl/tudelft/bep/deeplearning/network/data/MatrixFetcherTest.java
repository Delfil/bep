package nl.tudelft.bep.deeplearning.network.data;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;
import org.nd4j.linalg.dataset.api.DataSet;

public class MatrixFetcherTest {

	@Test
	public void test() {
		String testFile = "test_data/correct_1";
		Data data = Data.readDataSet(testFile);
		assertEquals(1, data.getBatchSize());
		MatrixDatasetIterator di = new MatrixDatasetIterator(data, 0.0, 1.0);
		read(di);
		assertFalse(di.hasNext());
		di.reset();
		read(di);
	}

	private static void read(final MatrixDatasetIterator di) {
		List<DataSet> results = new ArrayList<>();
		int count = 0;
		while (di.hasNext()) {
			results.add(di.next());
			count++;
		}
		assertEquals(3, count);
		assertEquals(3, results.size());
		for (int i = 0; i < 2; i++) {
			for (int j = i + 1; j < 3; j++) {
				assertNotEquals(results.get(i), results.get(j));
			}
		}

	}

	@Test(expected = IllegalStateException.class)
	public void emptyFetchTest() {
		String testFile = "test_data/correct_1";
		Data data = Data.readDataSet(testFile);
		MatrixDataFetcher df = new MatrixDataFetcher(data, 0.0, 1.0);
		MatrixDatasetIterator di = new MatrixDatasetIterator(data, df);
		df.fetch(4);
		assertFalse(di.hasNext());
		df.fetch(1);
	}
}
