/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.graphx.lib

import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.jblas.DoubleMatrix

import scala.util.Random

/**
 * Implementation of LDA which is optimized by gibbs sampling.
 */
object LDA {

  /** Configuration Parameters for LDA */
  class Conf (
               var topicNum : Int,
               var vocSize  : Int,
               var alpha    : Double,
               var beta     : Double,
               var maxIter  : Int)
    extends Serializable

  def run(edges: RDD[Edge[Int]], conf: Conf) :
    (Graph[DoubleMatrix, DoubleMatrix], DoubleMatrix, Double) = {

    //Generate default vertex attribute
    def defaultF(topicNum: Int): DoubleMatrix = {
      val v1 = new DoubleMatrix(topicNum)
      for(i <- 0 until topicNum) {
        v1.put(i, 0.0)
      }
      v1
    }

    //get a topic from multinomial distribution
    def getRandFromMultinomial(topicDist: Array[Double]): Int = {
      val rand = Random.nextDouble()
      val s = doubleArrayOps(topicDist).sum
      val arrNormalized = doubleArrayOps(topicDist).map{ e => e / s}
      var localSum = 0.0
      val cumArr = doubleArrayOps(arrNormalized).map{ dist =>
        localSum = localSum + dist
        localSum
      }
      doubleArrayOps(cumArr).indexWhere(cumDist => cumDist >= rand)
    }

    def initMapFunc(topicNum: Int, triplet: EdgeTriplet[DoubleMatrix, Int])
    : DoubleMatrix = {
      val tokenTopic = new DoubleMatrix(triplet.attr)
      for(i <- 0 until triplet.attr) {
        tokenTopic.put(i, Random.nextInt(topicNum).toDouble)
      }
      println("check-1: wordId: " + triplet.srcId + " docId: " + triplet.dstId
        + " topic: " + tokenTopic.get(0))
      tokenTopic
    }

    def vertexUpdateFunc(conf: Conf)
                        (et: EdgeTriplet[DoubleMatrix, DoubleMatrix])
    : Iterator[(VertexId, DoubleMatrix)] = {
      val termTopic = new DoubleMatrix(conf.topicNum)
      println("vertexUpdate: wordId: " + et.srcId + " docId: " + et.dstId
        + " topic: " + et.attr.get(0))
      for (i <- 0 until et.attr.length) {
        val t = et.attr.get(i).toInt
        termTopic.put(t, termTopic.get(t) + 1)
      }
      Iterator((et.srcId, termTopic), (et.dstId, termTopic))
    }

    def gibbsSampling(conf: Conf, topicWord: DoubleMatrix)
                   (et: EdgeTriplet[DoubleMatrix, DoubleMatrix])
    : DoubleMatrix = {
      println("gibbsSample: wordId: " + et.srcId + " docId: " + et.dstId
        + " topic: " + et.attr.get(0))
      val (doc, word) = (et.srcAttr, et.dstAttr)
      val tokenTopic = new DoubleMatrix(et.attr.length)
      for ( i <- 0 until et.attr.length) {
        val t = et.attr.get(i).toInt
        doc.put(t, doc.get(t) - 1)
        word.put(t, word.get(t) - 1)
        topicWord.put(t, topicWord.get(t) - 1)
        val topicDist = new Array[Double](conf.topicNum)
        for (k <- 0 until conf.topicNum) {
          topicDist(k) = ( (doc.get(k) + conf.alpha) * (word.get(k) + conf.beta)
            / (topicWord.get(k) + conf.vocSize * conf.beta) )
        }
        val newTopic = getRandFromMultinomial(topicDist)
        tokenTopic.put(i, newTopic)
        doc.put(newTopic, doc.get(newTopic) + 1)
        word.put(newTopic, word.get(newTopic) + 1)
        topicWord.put(newTopic, topicWord.get(newTopic) + 1)
      }
      tokenTopic
    }

    def computePerplexity(conf: Conf, topicWord: DoubleMatrix)
                         (et: EdgeTriplet[DoubleMatrix, DoubleMatrix])
    : Iterator[(VertexId, (Double, Double))] = {
      val (doc, word) = (et.srcAttr, et.dstAttr)
      val docDis = new DoubleMatrix(conf.topicNum)
      val wordDis = new DoubleMatrix(conf.topicNum)
      val nm = doc.sum()
      var sum = 0.0
      for ( i <- 0 until conf.topicNum ) {
        wordDis.put(i, (word.get(i) + conf.beta) / (topicWord.get(i) + conf.beta * conf.vocSize))
        docDis.put(i, (doc.get(i) + conf.alpha) / (nm + conf.topicNum * conf.alpha))
        sum += wordDis.get(i) * docDis.get(i)
      }
      val res = math.log(sum)
      Iterator((et.srcId, (res, nm)))
    }

    def computeDis(conf: Conf, topicWord: DoubleMatrix)
                  (et: EdgeTriplet[DoubleMatrix, DoubleMatrix])
    : Iterator[(VertexId, DoubleMatrix)] = {
      val (doc, word) = (et.srcAttr, et.dstAttr)
      val docDis = new DoubleMatrix(conf.topicNum)
      val wordDis = new DoubleMatrix(conf.topicNum)
      val nm = doc.sum()
      for ( i <- 0 until conf.topicNum ) {
        wordDis.put(i, (word.get(i) + conf.beta) / (topicWord.get(i) + conf.beta * conf.vocSize))
        docDis.put(i, (doc.get(i) + conf.alpha) / (nm + conf.topicNum * conf.alpha))
      }
      Iterator((et.srcId, docDis), (et.dstId, wordDis))
    }

    var perplexity : Double = 0.0

    edges.cache()
    val G = Graph.fromEdges(edges, defaultF(conf.topicNum))

    var g: Graph[DoubleMatrix, DoubleMatrix] = G.mapTriplets{
      (pid, iter) =>
        iter.map { triplet =>
          val tokenTopic = new DoubleMatrix(triplet.attr)
          for(i <- 0 until triplet.attr) {
            tokenTopic.put(i, Random.nextInt(conf.topicNum).toDouble)
          }
          println("check-1: wordId: " + triplet.srcId + " docId: " + triplet.dstId
              + " topic: " + tokenTopic.get(0))
          tokenTopic
        }
    }.cache()

    //debug
    for (triplet <- g.triplets.collect) {
      println("check1: wordId: " + triplet.srcId + " docId: " + triplet.dstId
        + " topic: " + triplet.attr.get(0))
    }

    //initialize topic for each token

    val t0 = g.mapReduceTriplets(
      vertexUpdateFunc(conf),
      (g1: DoubleMatrix, g2: DoubleMatrix) => g1.addColumnVector(g2)
    )

    //debug
    for (triplet <- g.triplets.collect) {
      println("check2: wordId: " + triplet.srcId + " docId: " + triplet.dstId
        + " topic: " + triplet.attr.get(0))
    }

    g = g.outerJoinVertices(t0) {
      (vid: VertexId, vd: DoubleMatrix,
       msg: Option[DoubleMatrix]) => msg.get
    }.cache()

    //debug
    for (triplet <- g.triplets.collect) {
      println("check3: wordId: " + triplet.srcId + " docId: " + triplet.dstId
        + " topic: " + triplet.attr.get(0))
    }

    for ( i <- 0 until conf.maxIter) {

      //compute total word number for each topic
      val topicWord = g.vertices.filter{ case (vid, vd) => vid % 2 == 1 }
        .map{ case (vid, vd) => vd}
        .reduce((a: DoubleMatrix, b: DoubleMatrix) => a.addColumnVector(b))

      //compute perplexity
      val perplexityG = g.mapReduceTriplets(
        computePerplexity(conf, topicWord),
        (g1: (Double, Double), g2: (Double, Double)) => (g1._1 + g2._1, g1._2)
      )

      //debug
      for (triplet <- g.triplets.collect) {
        println("check4: wordId: " + triplet.srcId + " docId: " + triplet.dstId
          + " topic: " + triplet.attr.get(0))
      }

      val perplexity_numerator = perplexityG.map{ case (vid, (res, nm)) =>
        if (vid %2 == 0) res else 0.0}.reduce(_ + _)

      val perplexity_denominator = perplexityG.map{ case (vid, (res, nm)) =>
        if (vid %2 == 0) nm else 0.0}.reduce(_ + _)

      perplexity = math.exp((perplexity_numerator / perplexity_denominator) * (-1.0))
      println("perplexity is: " + perplexity)

      val gibbsSample = (et: EdgeTriplet[DoubleMatrix, DoubleMatrix]) => {
        val (doc, word) = (et.srcAttr, et.dstAttr)
        val tokenTopic = new DoubleMatrix(et.attr.length)
        for ( i <- 0 until et.attr.length) {
          val t = et.attr.get(i).toInt
          doc.put(t, doc.get(t) - 1)
          word.put(t, word.get(t) - 1)
          topicWord.put(t, topicWord.get(t) - 1)
          val topicDist = new Array[Double](conf.topicNum)
          for (k <- 0 until conf.topicNum) {
            topicDist(k) = ( (doc.get(k) + conf.alpha) * (word.get(k) + conf.beta)
              / (topicWord.get(k) + conf.vocSize * conf.beta) )
          }
          val newTopic = getRandFromMultinomial(topicDist)
          tokenTopic.put(i, newTopic)
          doc.put(newTopic, doc.get(newTopic) + 1)
          word.put(newTopic, word.get(newTopic) + 1)
          topicWord.put(newTopic, topicWord.get(newTopic) + 1)
        }
        tokenTopic
      }

      //assign new topic for each token by using gibbsSampling
      g = g.mapTriplets{
        (pid, iter) =>
          iter.map {
            et =>
              gibbsSample(et)
          }
      }.cache()

      //debug
      for (triplet <- g.triplets.collect) {
        println("check5: wordId: " + triplet.srcId + " docId: " + triplet.dstId
          + " topic: " + triplet.attr.get(0))
      }

      //update N*T, M*T
      val t1 = g.mapReduceTriplets(
        vertexUpdateFunc(conf),
        (g1: DoubleMatrix, g2: DoubleMatrix) => g1.addColumnVector(g2)
      )

      //debug
      for (triplet <- g.triplets.collect) {
        println("check6: wordId: " + triplet.srcId + " docId: " + triplet.dstId
          + " topic: " + triplet.attr.get(0))
      }

      g = g.outerJoinVertices(t1) {
        (vid: VertexId, vd: DoubleMatrix,
         msg: Option[DoubleMatrix]) => msg.get
      }.cache()

      //debug
      for (triplet <- g.triplets.collect) {
        println("check7: wordId: " + triplet.srcId + " docId: " + triplet.dstId
          + " topic: " + triplet.attr.get(0))
      }

    }

    //compute total word number for each topic
    val topicWord = g.vertices.filter{ case (vid, vd) => vid % 2 == 1 }.map{ case (vid, vd) => vd
    }.reduce((a: DoubleMatrix, b: DoubleMatrix) => a.addColumnVector(b))

    //compute docTopicDis and wordTopicDis
    val disVer = g.mapReduceTriplets(
      computeDis(conf, topicWord),
      (g1: DoubleMatrix, g2: DoubleMatrix) => g1
    )
    val resG = g.outerJoinVertices(disVer) {
      (vid: VertexId, vd: DoubleMatrix,
       msg: Option[DoubleMatrix]) =>
       if (msg.isDefined) (msg.get) else vd
    }

    (resG, topicWord, perplexity)
  }
}
