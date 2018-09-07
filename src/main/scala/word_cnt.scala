/**
  * 使用SGD梯度下降法的逻辑回归
  * Created by wenjiahua on 2017/3/30 0030.
  */

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionWithSGD, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{DataFrame, SparkSession}

object word_cnt {
  //屏蔽不必要的日志显示在终端上
  Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
  Logger.getLogger("org.apache.eclipse.jetty.server").setLevel(Level.OFF)

  val sparkSession: SparkSession = SparkSession.builder
    .master("local")
    .appName("my-spark-app")
    .getOrCreate()

  def checkRDD(rdd: org.apache.spark.rdd.RDD[org.apache.spark.mllib.regression.LabeledPoint]): Boolean = {
    true
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
      r.getString(targetInd).toInt, // Get target value
      // Map feature indices to values
      Vectors.dense(featInd.map(r.getString(_).toDouble).toArray)
    )
    )
  }

  //  def LP2libSVM(rdd:RDD[LabeledPoint])

  def main(args: Array[String]): Unit = {
    //    val dataFrame = sparkSession.read.option("header","true").csv("./src/main/resouces/liushi.csv")
    //
    //    val schema =
    //      """uid,recentDate,dealPercent,transOutPercent,age,Rank,CreditCard,payCredit,cashHold,Salary,Channel,licai,fund,security,gold,sex,caifuchanpin,qianli,historicalArriveRate,Target""".split(",")
    //    val sm = StructType(schema.map{
    //      x=>
    //        StructField(x,if (x =="uid") StringType else StringType,nullable = true)
    //    })
    //
    //    val df = sparkSession.createDataFrame(dataFrame.rdd,sm)
    //    df.createOrReplaceTempView("liushi")
    //
    //    val data = stringRdd2LP(sparkSession.sql(
    //      """select * from liushi where Target in (0,1)""".stripMargin),Array("uid"),"Target")

    val data = MLUtils.loadLibSVMFile(sparkSession.sparkContext, "./src/main/resouces/data_input_libSVM.txt")
    val vectors = data.map(x => x.features)
    val rows = new RowMatrix(vectors)
    println("方差：", rows.computeColumnSummaryStatistics().variance)
    //每列的方差
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(vectors)
    //标准化
    val scaled_data = data.map(point => LabeledPoint(point.label, scaler.transform(point.features)))
      .randomSplit(Array(0.7, 0.3), 11L)
    val data_train = scaled_data(0)
    val data_test = scaled_data(1)
    val numIteration = 20

    //    val model_NBys=NaiveBayes.train(data_train,numIteration)
    //    val correct_NBys=data_test.map{
    //      point => if(model_NBys.predict(point.features)==point.label)
    //        1 else 0
    //    }.sum()/data_test.count()//精确度：0.6060885608856088
    //    println(correct_NBys.toString)
    //    val metrics_NBys=Seq(model_NBys).map{
    //      model =>
    //        val socreAndLabels=data_test.map {
    //          point => (model.predict(point.features), point.label)
    //        }
    //        val metrics=new BinaryClassificationMetrics(socreAndLabels)
    //        (model.getClass.getSimpleName,metrics.areaUnderPR(),metrics.areaUnderROC())
    //    }
    //    val allMetrics_NBys = metrics_NBys
    //    allMetrics_NBys.foreach{ case (m, pr, roc) =>
    //      println("NBys:"+f"$m, Area under PR: ${pr * 100.0}%2.4f%%, Area under ROC: ${roc * 100.0}%2.4f%%")
    //    }


    val model_lrSGD = LogisticRegressionWithSGD.train(data_train, numIteration)
    val correct_lrSGD = data_test.map {
      point =>
        if (model_lrSGD.predict(point.features) == point.label)
          1 else 0
    }.sum() / data_test.count() //精确度：0.6060885608856088
    println(correct_lrSGD.toString)
    val metrics_lrSGD = Seq(model_lrSGD).map {
      model =>
        val socreAndLabels = data_test.map {
          point => (model.predict(point.features), point.label)
        }
        val metrics = new BinaryClassificationMetrics(socreAndLabels)
        (model.getClass.getSimpleName, metrics.areaUnderPR(), metrics.areaUnderROC())
    }
    val allMetrics_lrSGD = metrics_lrSGD
    allMetrics_lrSGD.foreach { case (m, pr, roc) =>
      println("lrSGD:" + f"$m, Area under PR: ${pr * 100.0}%2.4f%%, Area under ROC: ${roc * 100.0}%2.4f%%")
    }


    val model_Svm = SVMWithSGD.train(data_train, numIteration)
    val correct_svm = data_test.map {
      point =>
        if (model_Svm.predict(point.features) == point.label)
          1 else 0
    }.sum() / data_test.count() //精确度：0.6060885608856088
    println(correct_svm.toString)
    val metrics_svm = Seq(model_Svm).map {
      model =>
        val socreAndLabels = data_test.map {
          point => (model.predict(point.features), point.label)
        }
        val metrics = new BinaryClassificationMetrics(socreAndLabels)
        (model.getClass.getSimpleName, metrics.areaUnderPR(), metrics.areaUnderROC())
    }
    val allMetrics_svm = metrics_svm
    allMetrics_svm.foreach { case (m, pr, roc) =>
      println("svm:" + f"$m, Area under PR: ${pr * 100.0}%2.4f%%, Area under ROC: ${roc * 100.0}%2.4f%%")
    }

    val model_lgR = new LogisticRegressionWithLBFGS().setNumClasses(2).run(data_train)
    val correct_lgR = data_test.map {
      point =>
        if (model_lgR.predict(point.features) == point.label)
          1 else 0
    }.sum() / data_test.count() //精确度：0.6060885608856088

    println(correct_lgR.toString)
    val metrics_lgR = Seq(model_lgR).map {
      model =>
        val socreAndLabels = data_test.map {
          point => (model.predict(point.features), point.label)
        }
        val metrics = new BinaryClassificationMetrics(socreAndLabels)
        (model.getClass.getSimpleName, metrics.areaUnderPR(), metrics.areaUnderROC())
    }
    val allMetrics_lgR = metrics_lgR
    allMetrics_lgR.foreach { case (m, pr, roc) =>
      println("lgR:" + f"$m, Area under PR: ${pr * 100.0}%2.4f%%, Area under ROC: ${roc * 100.0}%2.4f%%")
    }
    //    /**
    //      * 首先介绍一下 libSVM的数据格式
    //        Label 1:value 2:value ….
    //        Label：是类别的标识
    //        Value：就是要训练的数据，从分类的角度来说就是特征值，数据之间用空格隔开
    //        比如: -15 1:0.708 2:1056 3:-0.3333
    //      */
    //
    //    val splits = data.randomSplit(Array(0.6,0.4),seed = 11L) //对数据集切分成两部分，一部分训练模型，一部分校验模型
    //    //splits.foreach(println)
    //    val parsedData =splits(0)
    //    val parsedTest =splits(1)
    //
    //    val numiteartor = 50
    //    val model = LogisticRegressionWithSGD.train(parsedData,numiteartor) //训练模型
    //    println(model.weights)
    //
    //    val predictionAndLabels = parsedTest.map{                           //计算测试值
    //      case LabeledPoint(label,features) =>
    //        (model.predict(features),label)                                              //存储测试值和预测值
    //    }                                                                   //rdd.map(x=> x+1)在这里先写x，此处是一个case class
    //    // 而后是对x的操作，即定义了一个prediction来存预测的结
    //    // 果，然后再定义返回值
    //    predictionAndLabels.foreach(println)
    //    val trainErr = predictionAndLabels.filter( r => r._1 != r._2).count.toDouble / parsedTest.count
    //    println("容错率为trainErr： " +trainErr)
    //    /**
    //      * 容错率为trainErr： 0.3
    //      */
    //
    //    val metrics = new MulticlassMetrics(predictionAndLabels)           //创建验证类
    //    val precision = metrics.accuracy                                   //计算验证值
    //    println("Precision= "+precision)

    //    val patient = Vectors.dense(Array(70,3,180.0,4,3)) //计算患者可能性
    //
    //
    //
    //    if(model.predict(patient) == 1)println("患者的胃癌有几率转移。 ")
    //    else println("患者的胃癌没有几率转移 。")
    //    /**
    //      * Precision= 0.7
    //      患者的胃癌没有几率转移
    //      */
  }
}