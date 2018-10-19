package com.scala.featureSelect

import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession

object DecisionTreeModel {
  def main(args: Array[String]): Unit = {
    val sparkSession = SparkSession.builder
      .master("local")
      .appName("my-spark-app")
      .getOrCreate()
    val sc = sparkSession.sparkContext

//    val orig_file=sc.textFile("train_nohead.tsv")
    //println(orig_file.first())
    val data_file=MLUtils.loadLibSVMFile(sc, "./src/main/resouces/liushi2_libsvm") //设置数据集
//    val data_file=MLUtils.loadLibSVMFile(sc, "./src/main/resouces/data_input_libSVM.txt") //设置数据集
    /*特征标准化优化，似乎对决策数没啥影响*/
    val vectors=data_file.map(x =>x.features)
    val rows=new RowMatrix(vectors)
    println(rows.computeColumnSummaryStatistics().variance)//每列的方差
    val scaler=new StandardScaler(withMean=true,withStd=true).fit(vectors)//标准化
    val scaled_data=data_file.map(point => LabeledPoint(point.label,scaler.transform(point.features)))
      .randomSplit(Array(0.7,0.3),11L)//固定seed为11L，确保每次每次实验能得到相同的结果
    val data_train=scaled_data(0)
    val data_test=scaled_data(1)

    val maxTreeDepth = 4


    /*训练决策树模型*/
    val model_DT=DecisionTree.train(data_train,Algo.Classification,Entropy,maxTreeDepth)
    /*决策树的精确度*/
    val predectionAndLabeledDT=data_test.map { point =>
      val predectLabeled = if (model_DT.predict(point.features) > 0.5) 1.0 else 0.0
      (predectLabeled,point.label)
    }
    val metricsDT=new MulticlassMetrics(predectionAndLabeledDT)
    println(metricsDT.accuracy)//0.6273062730627307
    /*决策树的PR曲线和AOC曲线*/
    val dtMetrics = Seq(model_DT).map{ model =>
      val scoreAndLabels = data_test.map { point =>
        val score = model.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
    }
    val allMetrics = dtMetrics
    allMetrics.foreach{ case (m, pr, roc) =>
      println(f"$m, Area under PR: ${pr * 100.0}%2.4f%%, Area under ROC: ${roc * 100.0}%2.4f%%")
    }
  }

}
