package com.scala.featureSelect

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}


/**
  * Created by legotime on 2016/4/8.
  */
object Correlations {
  case class pair (featureName:String,corrScore:Double)
  def main(args: Array[String]) {
    //    val sparkConf = new SparkConf().setAppName("Correlations").setMaster("local")
    //    val sc = new SparkContext(sparkConf)
    //    val rdd1 = sc.parallelize(
    //      Array(
    //        Array(1.0, 2.0, 3.0, 4.0),
    //        Array(2.0, 3.0, 4.0, 5.0),
    //        Array(3.0, 4.0, 5.0, 6.0)
    ////      )
    //    ).map(f => Vectors.dense(f))
    val sparkSession = SparkSession.builder
      .master("local")
      .appName("my-spark-app")
      .getOrCreate()
    val sc = sparkSession.sparkContext
    val dataFrame = sparkSession.read.option("header", "true").csv("./src/main/resouces/liushi.csv")

    val schema =
      """uid,recentDate,dealPercent,transOutPercent,age,Rank,CreditCard,payCredit,cashHold,Salary,Channel,licai,fund,security,gold,sex,caifuchanpin,qianli,historicalArriveRate,Target""".split(",")
    val sm = StructType(schema.map {
      x => StructField(x, if (x == "uid") StringType else StringType, nullable = true)
    })

    val df = sparkSession.createDataFrame(dataFrame.rdd, sm)
    df.createOrReplaceTempView("liushi")

    val data = stringRdd2LP(sparkSession.sql(
      """select * from liushi where Target in (0,1)""".stripMargin), Array("uid"), "Target")
    //    println(data.take(2))
    val featuresRdd = data.map(line => Vectors.dense(line.features.toArray))
    //package org.apache.spark.mllib.stat下的一些基本操作
    val summary = Statistics.colStats(featuresRdd)
    println(summary.mean) // a dense vector containing the mean value for each column每列的平均值
    println(summary.variance) // column-wise variance,列方差
    println(summary.numNonzeros) // number of nonzeros in each column每列非零个数
    println(summary.normL2) //二范数
    println(summary.max) //每列最大

    //==================================计算相关性系数===================================
    //val rdd2 = sc.parallelize(Array(1.0,2.0,3.0,4.0))
    //val rdd3 = sc.parallelize(Array(2.0,3.0,4.0,5.0))

    val baseDf = sparkSession.sql(
      """select * from liushi where Target in (0,1)""".stripMargin)
    val target_ary = stringRdd2ArrayByCol(baseDf, "Target")
    val feature_ary = stringRdd2ArrayByCol(baseDf, "caifuchanpin")

    //    val xx = target_rdd.foreach(_.toArray)
    //    println("xxxxxx"+xx.toString)

//    val rdd2 = sc.parallelize(Array(161.0, 176.0, 174.0, 198.0, 182.0, 178.0, 190.0, 180.0))
//    val rdd3 = sc.parallelize(Array(81.0, 88.0, 87.0, 99.0, 91.0, 89.0, 95.0, 90.0))
//    println("rdd2:" + rdd2)
//    rdd2.collect().foreach(println)
//    val correlation1:Double = Statistics.corr(rdd2, rdd3, "pearson")
    //缺省的情况下，默认的是pearson相关性系数
    val correlation1: Double = Statistics.corr(sc.parallelize(target_ary), sc.parallelize(feature_ary))
    println("pearson相关系数：" + correlation1)
    //    //pearson相关系数：0.6124030566141675
    //    val correlation2: Double = Statistics.corr(rdd2, rdd3, "spearman")
    //    println("spearman相关系数：" + correlation2)
    //    //spearman相关系数：0.7395161835775294
    val featureCorr = getFeaturesCorr(sparkSession,"liushi",schema,"Target")
    featureCorr.foreach(println(_))
    sc.stop()
  }

  def stringRdd2LP(df: DataFrame, ignoreCols: Array[String], targetCol: String)
  : org.apache.spark.rdd.RDD[org.apache.spark.mllib.regression.LabeledPoint] = {
    // Map feature names to indices
    val ignored = ignoreCols.map(df.columns.indexOf(_))

    // Or if you want to exclude columns
    //    val ignored = List("foo", "target", "x2")
    //    val featInd = df.columns.foldLeft(ignored)(_.diff(_))
    val featInd = df.columns.diff(ignoreCols).map(df.columns.indexOf(_))

    // Get index of target
    val targetInd = df.columns.indexOf(targetCol)

    df.rdd.map(r => LabeledPoint(
      r.getString(targetInd).toDouble, // Get target value
      // Map feature indices to values
      Vectors.dense(featInd.map(r.getString(_).toDouble))
    )
    )
  }

  def stringRdd2ArrayByCol(df: DataFrame, selectCol: String): Array[Double] = {
    df.rdd.collect().map(r => r.getString(df.columns.indexOf(selectCol)).toDouble)
  }

  def getFeaturesCorr(sparkSession:SparkSession,tempView:String,schema:Array[String],Target:String) = {
    val sc =sparkSession.sparkContext
    val baseDF = sparkSession.sql(
      """select * from """+tempView+""" where Target in (0,1)""".stripMargin)
    val target = stringRdd2ArrayByCol(baseDF, Target)
    val features = schema.diff(Array(Target,"uid")).map(x => stringRdd2ArrayByCol(baseDF, x))

    val result = features.map {
      x=>
      Statistics.corr(sc.parallelize(target), sc.parallelize(x))

    }
    result
//    schema.diff(Target).map(Statistics.corr(sc.parallelize(target), sc.parallelize(stringRdd2ArrayByCol(
//      sparkSession.sql(
//        """select * from """+tempView+""" where Target in (0,1)""".stripMargin
//      ), _)))
  }
}