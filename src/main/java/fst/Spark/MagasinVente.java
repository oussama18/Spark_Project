package fst.Spark;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;

import java.util.Arrays;

public class MagasinVente {
    private static final Logger LOGGER = LoggerFactory.getLogger(MagasinVente.class);

    public static void main(String[] args) {
        new MagasinVente().run(args[0], args[1]);
    }

    public void run(String inputFilePath, String outputDir) {
        String master = "local[*]";
        SparkConf conf = new SparkConf()
                .setAppName(MagasinVente.class.getName())
                .setMaster(master)
                ;
        JavaSparkContext sc = new JavaSparkContext(conf);

        JavaRDD<String> textFile = sc.textFile(inputFilePath);

        JavaPairRDD<String, Double> totalVente = textFile.
                map(s -> Arrays.asList(s.split("\t")))
                .mapToPair(list -> new Tuple2<>(
                        list.get(2),
                        Double.parseDouble(list.get(4))
                        )
                )
                .reduceByKey((a, b) -> a + b);
        totalVente.saveAsTextFile(outputDir);
    }
}
