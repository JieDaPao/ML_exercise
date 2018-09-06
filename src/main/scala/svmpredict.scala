/**
  *使用SGD梯度下降法的逻辑回归
  * Created by wenjiahua on 2017/3/30 0030.
  */
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, _}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
object svmpredict{
  //屏蔽不必要的日志显示在终端上
  Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
  Logger.getLogger("org.apache.eclipse.jetty.server").setLevel(Level.OFF)

  def classify(trainData:RDD[org.apache.spark.mllib.regression.LabeledPoint],
               testData:RDD[org.apache.spark.mllib.regression.LabeledPoint],
               numiteartor:Int):Unit = {
//    val svm = method  .train(trainData,numiteartor)
//    println(svm.weights)
//    val predictionAndLabels_lgR = testData.map{                           //计算测试值
//      case LabeledPoint(label,features) =>
//        val prediction = lgR.predict(features)
//        (prediction,label)                                              //存储测试值和预测值
//    }
//    //    predictionAndLabels_lgR.foreach(println)
//    val trainErr_lgR = predictionAndLabels_lgR.filter( r => r._1 != r._2).count.toDouble / parsedTest.count
//    println("lgR容错率为trainErr： " +trainErr_lgR)
  }

  def main(args: Array[String]) {

    val conf = new SparkConf().setMaster("local[1]").setAppName("svmpredict")
    val sc = new SparkContext(conf)

    val data = MLUtils.loadLibSVMFile(sc,"./src/main/resouces/iris_libsvm.txt")
    val scaled_data=data
      .randomSplit(Array(0.7,0.3),11L)
    val data_train=scaled_data(0)
    val data_test=scaled_data(1)


    val numIterations = 20
    val model=new LogisticRegressionWithLBFGS().setNumClasses(2).run(data_train)

    val labelAndPreds = data.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)

    }
    val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / data.count
    println("Training Error = " + trainErr)
  }
}