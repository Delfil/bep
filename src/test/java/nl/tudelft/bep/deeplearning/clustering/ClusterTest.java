package nl.tudelft.bep.deeplearning.clustering;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;

import nl.tudelft.bep.deeplearning.clustering.Cluster;
import nl.tudelft.bep.deeplearning.clustering.exception.MinimumNotPossibleException;

public class ClusterTest {
	
	Cluster point1 = new Cluster(1,2,0);
	Cluster point2 = new Cluster(4,5,1);
	Cluster point3 = new Cluster(1,3,2);
	Cluster point4 = new Cluster(4,6,3);
	
	Cluster layer1_1 = new Cluster(1,2.5,0);
	Cluster layer1_2 = new Cluster(4,5.5,1);
	
	Cluster root = new Cluster(0);

	@Before
	public void setup() {

		layer1_1.addCluster(point1);
		layer1_1.addCluster(point3);
		layer1_2.addCluster(point2);
		layer1_2.addCluster(point4);
		
		root.addCluster(layer1_1);
		root.addCluster(layer1_2);
	}

	@Test
	public void testIndices() {	
		ArrayList<Integer> expect1_1 = new ArrayList<Integer>();
		expect1_1.add(0);
		expect1_1.add(2);
		ArrayList<Integer> expect1_2 = new ArrayList<Integer>();
		expect1_2.add(1);
		expect1_2.add(3);
		
		ArrayList<Integer> expect_root = new ArrayList<Integer>();
		expect_root.add(0);
		expect_root.add(2);
		expect_root.add(1);
		expect_root.add(3);
		
		assertEquals(expect1_1, layer1_1.listIndices());
		assertEquals(expect1_2, layer1_2.listIndices());
		assertEquals(expect_root, root.listIndices());
	}
	
	@Test
	public void testSize() {
		assertEquals(2, layer1_1.size());
		assertEquals(2, layer1_2.size());
		assertEquals(4, root.size());
		assertEquals(1, point1.size());
	}
	
	@Test
	public void testEquals() {
		Cluster root_same = new Cluster(0);
		assertFalse(root.equals(root_same));
		
		root_same.addCluster(layer1_1);
		root_same.addCluster(layer1_2);
		root.calculateMean();
		root_same.calculateMean();
		assertTrue(root.equals(root_same));
		
		int wrongType = 0;
		assertFalse(root.equals(wrongType));

	}	
	
	@Test 
	public void testLayerSize() throws FileNotFoundException, IOException, MinimumNotPossibleException {
		Cluster[] input = Mapping.read(new FileInputStream(new ClassPathResource("points.in").getFile()));
		Cluster[] result = Mapping.createClusters(input);
		result = Mapping.createClusters(result);	
		
		assertEquals(1, result[0].layerSize(0));
		assertEquals(2, result[0].layerSize(1));
		assertEquals(4, result[0].layerSize(2));
		
	}
	
	@Test 
	public void testLayer() throws FileNotFoundException, IOException, MinimumNotPossibleException {		
		Cluster point1 = new Cluster(1,2,0);
		Cluster point2 = new Cluster(4,5,1);
		Cluster point3 = new Cluster(1,3,2);
		Cluster point4 = new Cluster(4,6,3);
		
		Cluster layer1_1 = new Cluster(0);
		Cluster layer1_2 = new Cluster(1);
		
		layer1_1.addCluster(point1);
		layer1_1.addCluster(point3);
		layer1_2.addCluster(point2);
		layer1_2.addCluster(point4);
		
		Cluster root = new Cluster(0);
		root.addCluster(layer1_1);
		root.addCluster(layer1_2);
		
		
		ArrayList<Cluster> expectLayer1 = new ArrayList<Cluster>(2);
		expectLayer1.add(layer1_1);
		expectLayer1.add(layer1_2);
		
		ArrayList<Cluster> expectLayer2 = new ArrayList<Cluster>(4);
		expectLayer2.add(point1);
		expectLayer2.add(point3);
		expectLayer2.add(point2);
		expectLayer2.add(point4);
		
		assertEquals(expectLayer1, root.layer(1));
		assertEquals(expectLayer2, root.layer(2));
		
	}
}
