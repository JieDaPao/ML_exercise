import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession

object svm_new {
  def main(args: Array[String]): Unit = {
    //println(orig_file.first())
    val spark = SparkSession.builder
      .master("local")
      .appName("my-spark-app")
      .getOrCreate()
    val data_file = MLUtils.loadLibSVMFile(spark.sparkContext, "./src/main/resouces/sample_svm_data.txt")

      /*特征标准化优化*/
      //    val vectors=data_file.map(x =>x.features)
      //    val rows=new RowMatrix(vectors)
      //    println(rows.computeColumnSummaryStatistics().variance)//每列的方差
      //    val scaler=new StandardScaler(withMean=true,withStd=true).fit(vectors)//标准化
      //    val scaled_data=data_file.map(point => LabeledPoint(point.label,scaler.transform(point.features)))
      .randomSplit(Array(0.7, 0.3), 11L)
    val data_train = data_file(0)
    val data_test = data_file(1)

    val numiteartor = 50
    /*训练决策树模型*/
    val model_DT = SVMWithSGD.train(data_train, numiteartor)
    //    val model_DT=DecisionTree.train(data_train,Algo.Classification,Entropy,maxDepth = 6)
    /*决策树的精确度*/
    val predectionAndLabeledDT = data_test.map { point =>
      val predectLabeled = if (model_DT.predict(point.features) > 0.5) 1.0 else 0.0
      (predectLabeled, point.label)
    }
    val metricsDT = new MulticlassMetrics(predectionAndLabeledDT)
    println(metricsDT.accuracy)
    //0.6273062730627307
    /*决策树的PR曲线和AOC曲线*/
    val dtMetrics = Seq(model_DT).map { model =>
      val scoreAndLabels = data_test.map { point =>
        val score = model.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
    }
    val allMetrics = dtMetrics
    allMetrics.foreach { case (m, pr, roc) =>
      println(f"$m, Area under PR: ${pr * 100.0}%2.4f%%, Area under ROC: ${roc * 100.0}%2.4f%%")
    }
    /*
    DecisionTreeModel, Area under PR: 74.2335%, Area under ROC: 62.4326%
    */
  }

}
