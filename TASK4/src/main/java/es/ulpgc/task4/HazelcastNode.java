package es.ulpgc.task4;

import com.hazelcast.config.Config;
import com.hazelcast.core.Hazelcast;
import com.hazelcast.core.HazelcastInstance;

public class HazelcastNode {

    private static HazelcastInstance instance;

    public static synchronized HazelcastInstance startNode() {
        if (instance == null) {
            Config config = new Config();
            config.setClusterName("matrix-cluster");
            instance = Hazelcast.newHazelcastInstance(config);
        }
        return instance;
    }

    public static HazelcastInstance getInstance() {
        return instance;
    }
}
