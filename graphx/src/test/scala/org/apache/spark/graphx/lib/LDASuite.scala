package org.apache.spark.graphx.lib

import org.apache.spark.graphx.{Edge, LocalSparkContext}
import org.scalatest.FunSuite

/**
 * Created by darlwen on 14-9-16.
 */
class LDASuite extends FunSuite with LocalSparkContext{

  test("Test LDA with perplexity on training set") {
    withSpark{ sc =>
      val threshold = 5
      val edges = sc.textFile(getClass.getResource("lda-test.data").getFile).map { line =>
        val fields = line.split(",")
        Edge(fields(0).toLong * 2, fields(1).toLong * 2 + 1, fields(2).toInt)
      }
      val conf = new LDA.Conf(10, 6, 0.1, 0.1, 5)
      var (graph, topicWord, perplexity) = LDA.run(edges, conf)

      assert(perplexity <= threshold)

    }
  }

}
