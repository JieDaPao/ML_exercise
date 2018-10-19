/**
  * 使用SGD梯度下降法的逻辑回归
  * Created by wenjiahua on 2017/3/30 0030.
  */

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

object LogisticRegression3 {
  //屏蔽不必要的日志显示在终端上
  Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
  Logger.getLogger("org.apache.eclipse.jetty.server").setLevel(Level.OFF)

  val conf = new SparkConf().setMaster("local").setAppName("word_cnt") //设置环境变变量
  val sc = new SparkContext(conf)

  def main(args: Array[String]): Unit = {
    val data = MLUtils.loadLibSVMFile(sc, "./src/main/resouces/liushi2_libsvm") //设置数据集
    /**
      * 首先介绍一下 libSVM的数据格式
      * Label 1:value 2:value ….
      * Label：是类别的标识
      * Value：就是要训练的数据，从分类的角度来说就是特征值，数据之间用空格隔开
      * 比如: -15 1:0.708 2:1056 3:-0.3333
      */
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L) //对数据集切分成两部分，一部分训练模型，一部分校验模型
    //splits.foreach(println)
    val parsedData = splits(0)
    val parsedTest = splits(1)

    val numiteartor = 50
    val model = LogisticRegressionWithSGD.train(parsedData, numiteartor) //训练模型
    println(model.weights)

    val predictionAndLabels = parsedTest.map { //计算测试值
      case LabeledPoint(label, features) =>
        (model.predict(features), label) //存储测试值和预测值
    }
    // 而后是对x的操作，即定义了一个prediction来存预测的结
    // 果，然后再定义返回值
//    predictionAndLabels.foreach(println)
    val trainErr = predictionAndLabels.filter(r => r._1 != r._2).count.toDouble / parsedTest.count
    println("容错率为trainErr： " + trainErr)
    /**
      * 容错率为trainErr： 0.3
      */

    val metrics = new MulticlassMetrics(predictionAndLabels) //创建验证类
    val precision = metrics.accuracy //计算验证值
    println("Precision= " + precision)

    val patient = Vectors.dense(Array(70, 3, 180.0, 4, 3)) //计算患者可能性


    if (model.predict(patient) == 1) println("患者的胃癌有几率转移。 ")
    else println("患者的胃癌没有几率转移 。")

    /**
      * Precision= 0.7
      * 患者的胃癌没有几率转移
      */
  }
}