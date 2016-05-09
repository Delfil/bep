package nl.tudelft.bep.deeplearning.clustering;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;

import org.junit.Test;

public class ClusterTest {
	
	Cluster point1 = new Cluster(1,2,0);
	Cluster point2 = new Cluster(4,5,1);
	Cluster point3 = new Cluster(1,3,2);
	Cluster point4 = new Cluster(4,6,3);
	
	Cluster layer1_1 = new Cluster(1,2.5,0);
	Cluster layer1_2 = new Cluster(4,5.5,1);
	
	Cluster root = new Cluster(0);
	
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
		setup();
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
		setup();
		assertEquals(2, layer1_1.size());
		assertEquals(2, layer1_2.size());
		assertEquals(4, root.size());
		assertEquals(1, point1.size());
	}
	
	@Test
	public void testEquals() {
		setup();
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
	public void testClone() {
		setup();
		Cluster root_same = root.clone();
		assertEquals(root, root_same);
	}
	
	
	
}
