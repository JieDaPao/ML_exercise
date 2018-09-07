package com.lxw1234.spark.features.selectors

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.ml.linalg.Vectors

/**
  * By http://lxw1234.com
  */
object TestChiSqSelector extends App {
  val conf = new SparkConf().setMaster("local").setAppName("lxw1234.com")
  val sc = new SparkContext(conf)
  val sqlContext = new org.apache.spark.sql.SQLContext(sc)

  import sqlContext.implicits._

  //构造数据集
  val data = Seq(
    (7, Vectors.dense(0.0, 0.0, 18.0, 1.0), 1.0),
    (8, Vectors.dense(0.0, 1.0, 12.0, 0.0), 0.0),
    (9, Vectors.dense(1.0, 0.0, 15.0, 0.1), 0.0)
  )
  val df = sc.parallelize(data).toDF("id", "features", "clicked")
  df.select("id", "features", "clicked").show()
  //使用卡方检验，将原始特征向量（特征数为4）降维（特征数为3）
  val selector = new ChiSqSelector().setNumTopFeatures(3).setFeaturesCol("features").setLabelCol("clicked").setOutputCol("selectedFeatures")
  val result = selector.fit(df).transform(df)
  result.show()

}