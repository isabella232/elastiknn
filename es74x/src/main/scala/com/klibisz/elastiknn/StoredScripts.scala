package com.klibisz.elastiknn

import java.util
import java.util.Collections

import org.elasticsearch.action.admin.cluster.storedscripts.PutStoredScriptRequest
import org.elasticsearch.common.bytes.BytesArray
import org.elasticsearch.common.xcontent.XContentType
import org.elasticsearch.script.{Script, ScriptType, StoredScriptSource}

object StoredScripts {

  final case class ExactScript(id: String, scriptSource: StoredScriptSource) {
    val putRequest: PutStoredScriptRequest = new PutStoredScriptRequest(id, "score", new BytesArray("{}"), XContentType.JSON, scriptSource)
    def script(fieldProc: String, b: Array[Double]): Script = new Script(
      ScriptType.STORED,
      null,
      id,
      util.Map.of("fp", s"$fieldProc.exact.vector", "b", b)
    )
  }

  private val dummyScript = new StoredScriptSource(
    "painless",
    """
      |return 0.0;
      |""".stripMargin,
    Collections.emptyMap()
  )

  val exactL1: ExactScript = ExactScript("elastiknn-exact-l1", dummyScript)
  val exactL2: ExactScript = ExactScript("elastiknn-exact-l2", dummyScript)

  val exactAngular: ExactScript = ExactScript(
    "elastiknn-exact-angular",
    new StoredScriptSource(
      "painless",
      """
        |def a = doc[params.fp];
        |def b = params.b;
        |double dotprod = 0.0; // Dot product a and b.
        |double asqsum = 0.0;  // Squared sum of a.
        |double bsqsum = 0.0;  // Squared sum of b.
        |for (int i = 0; i < a.length; i++) {
        |  dotprod += a[i] * b[i];
        |  asqsum += Math.pow(a[i], 2);
        |  bsqsum += Math.pow(b[i], 2);
        |}
        |double sim = dotprod / (Math.sqrt(asqsum) * Math.sqrt(bsqsum));
        |return 1.0 + sim; // Can't have negative scores.
        |""".stripMargin,
      Collections.emptyMap()
    )
  )

  val exactHamming: ExactScript = ExactScript("elastiknn-exact-hamming", dummyScript)
  val exactJaccard: ExactScript = ExactScript("elastiknn-exact-jaccard", dummyScript)

  val exactScripts: Seq[ExactScript] = Seq(exactL1, exactL2, exactAngular, exactHamming, exactJaccard)

}