package com.elastiknn.annb

import akka.NotUsed
import akka.http.scaladsl.Http
import akka.http.scaladsl.client.RequestBuilding.Get
import akka.http.scaladsl.model.{HttpResponse, StatusCodes, Uri}
import akka.stream.scaladsl.{FileIO, Flow, Keep, Sink, Source}
import com.elastiknn.annb.Runner.Params
import com.klibisz.elastiknn.api.Vec
import io.circe.{Json, JsonObject}
import org.bytedeco.hdf5.global.hdf5._

import java.nio.file.Path
import scala.concurrent.Future

trait DatasetStore[V <: Vec.KnownDims] {

  /**
    * Provide an akka-stream Source for vectors that should be indexed.
    * If the dataset is not cached in local storage already, this should also download it.
    */
  def indexVectors(): Source[V, NotUsed]

  /**
    * Provide an akka-stream Source for vectors that should be used for querying.
    * If the dataset is not cached in local storage already, this should also download it.
    */
  def queryVectors(): Source[V, NotUsed]

  def saveResults(params: Params, fileName: String): Sink[LuceneResult, Future[NotUsed]]

}

object DatasetStore {

  /**
    * Client that reads datasets in the ann-benchmarks HDF5 format.
    * Each dataset is in a single HDF5 file at http://ann-benchmarks.com/<dataset-name>.hdf5
    * e.g., http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5
    */
  final class AnnBenchmarksDenseFloat(
      dataset: Dataset[Benchmark.AnnBenchmarks.type, Vec.DenseFloat],
      algo: Algorithm,
      datasetsPath: Path,
      resultsPath: Path,
      count: Int
  ) extends DatasetStore[Vec.DenseFloat] {

    private val datasetHdf5Path = datasetsPath.resolve(s"${dataset.name}.hdf5")
    private val resultsPrefixPath = resultsPath.resolve(dataset.name).resolve(count.toString).resolve(algo.name)

    private def download() =
      Source
        .fromMaterializer {
          case (mat, _) =>
            implicit val exc = mat.executionContext
            implicit val sys = mat.system
            val log = mat.system.log
            if (datasetHdf5Path.toFile.exists()) Source.single(())
            else
              Source.lazyFuture { () =>
                val uri = Uri(s"http://ann-benchmarks.com/${dataset.name}.hdf5")
                val req = Get(uri)
                val resF = Http().singleRequest(req)
                resF.flatMap {
                  case HttpResponse(StatusCodes.OK, _, entity, _) =>
                    log.info(s"Downloading dataset ${dataset.name} from $uri to $datasetHdf5Path")
                    entity.dataBytes
                      .runWith(FileIO.toPath(datasetHdf5Path))
                      .map(_ => log.info(s"Finished downloading dataset ${dataset.name} to $datasetHdf5Path"))
                  case other => Future.failed(new Throwable(s"Non-200 response: $other"))
                }
              }
        }
        .mapMaterializedValue(_ => NotUsed)

    private def readVectors(dataset: String): Source[Vec.DenseFloat, NotUsed] = {
      val t = HDF5Util.readFloats2d(datasetHdf5Path, H5F_ACC_RDONLY, dataset).map(_.map(Vec.DenseFloat(_)))
      Source
        .future(Future.fromTry(t))
        .flatMapConcat(i => Source.fromIterator(() => i))
    }

    override def indexVectors(): Source[Vec.DenseFloat, NotUsed] =
      download().flatMapConcat(_ => readVectors("train"))

    override def queryVectors(): Source[Vec.DenseFloat, NotUsed] =
      download().flatMapConcat(_ => readVectors("test"))

    override def saveResults(params: Params, fileName: String): Sink[LuceneResult, Future[NotUsed]] = {
      Flow
        .fromMaterializer {
          case (mat, _) =>
            Flow[LuceneResult]
              .fold(Vector.empty[LuceneResult])(_ :+ _)
              .mapAsync(1) { results: Vector[LuceneResult] =>
                Future.fromTry {
                  val fileNameWithHdf5 = if (fileName.endsWith(".hdf5")) fileName else s"$fileName.hdf5"
                  val fileNameNoHdf5 = if (fileNameWithHdf5 != fileName) fileName else fileName.dropRight(5)
                  val hdf5Path = resultsPrefixPath.resolve(fileNameWithHdf5)
                  mat.system.log.info(s"Writing results to [$hdf5Path]")
                  val bestSearchTime = results.map(_.time.toNanos / 1e9).sum * 1f / results.length
                  val candidates = results.map(_.neighbors.count(_ >= 0)).sum / results.length
                  for {
                    _ <- HDF5Util.createFileWithAttributes(
                      hdf5Path,
                      JsonObject(
                        "algo" -> Json.fromString(params.algo.name),
                        "batch_mode" -> Json.fromBoolean(params.batch),
                        "best_search_time" -> Json.fromDoubleOrNull(bestSearchTime),
                        "build_time" -> Json.fromFloatOrNull(-1f),
                        "candidates" -> Json.fromInt(candidates),
                        "count" -> Json.fromInt(params.count),
                        "dataset" -> Json.fromString(params.dataset.name),
                        "distance" -> Json.fromString(params.algo.distance),
                        "expect_extra" -> Json.fromBoolean(false),
                        "index_size" -> Json.fromFloatOrNull(-1f),
                        "name" -> Json.fromString(s"${params.algo.name}_$fileNameNoHdf5"),
                        "run_count" -> Json.fromInt(params.runs)
                      )
                    )
                    _ <- HDF5Util.writeFloats2d(hdf5Path, H5F_ACC_RDWR, "distances", results.map(_.distances).toArray)
                    _ <- HDF5Util.writeInts2d(hdf5Path, H5F_ACC_RDWR, "neighbors", results.map(_.neighbors).toArray)
                    _ <- HDF5Util.writeFloats1d(hdf5Path, H5F_ACC_RDWR, "times", results.map(_.time.toNanos / 1e9).map(_.toFloat).toArray)
                  } yield NotUsed
                }
              }
        }
        .toMat(Sink.head)(Keep.right)
    }
  }
}
