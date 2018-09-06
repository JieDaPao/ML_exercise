package com.scala.featureSelect

import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession


object ChiSqSelectorTest {

  case class Bean(id:Double,features:org.apache.spark.ml.linalg.Vector,clicked:Double){}

  def main(args: Array[String]) {
    val sparkSession = SparkSession.builder
      .master("local")
      .appName("my-spark-app")
      .getOrCreate()

//    val data_ori = MLUtils.loadLibSVMFile(sparkSession.sparkContext,"./src/main/resouces/data_input_libSVM.txt")

    val data_ori = sparkSession.read.format("libsvm").load("/Users/yujieshen/IdeaProjects/scala/ML_exercise/word_cnt/src/main/resouces/data_input_libSVM.txt")
//    val data_ori = MLUtils.loadLibSVMFile(sparkSession.sparkContext,"./src/main/resouces/sample_svm_data.txt")

    val sc = sparkSession.sparkContext
    sc.setLogLevel("WARN")
    var sqlContext = sparkSession.sqlContext

//    var id = .0
//    val data = data_ori.map{
//      id +=1.0
//      line => Seq(id,line.features,line.label)
//    }
//    val data_seq = data.collect().toSeq

//    val data =
//    Seq(
//          (7, Vectors.dense(0.0, 0.0, 18.0, 1.0), 1.0),
//          (8, Vectors.dense(0.0, 1.0, 12.0, 0.0), 0.0),
//          (9, Vectors.dense(1.0, 0.0, 15.0, 0.1), 0.0)
//        )
//        val data = MLUtils.loadLibSVMFile(sc,"./src/main/resouces/sample_svm_data.txt")

    val rdd = data_ori.rdd
    var id = .0
    val beanRDD = rdd.map{
          id +=1.0
          line => Bean(id,line.getAs("features"),line.getAs("label"))
        }
    val df = sqlContext.createDataFrame(beanRDD)

    val selector = new ChiSqSelector()
          .setNumTopFeatures(2)
          .setFeaturesCol("features")
          .setLabelCol("clicked")
          .setOutputCol("selectedFeatures")

    val result = selector.fit(df).transform(df)
        result.show()
  }



}
