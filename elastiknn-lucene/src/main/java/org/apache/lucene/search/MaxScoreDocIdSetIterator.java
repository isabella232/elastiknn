package org.apache.lucene.search;

import com.klibisz.elastiknn.search.HitCounter;

import java.io.IOException;

public class MaxScoreDocIdSetIterator extends DocIdSetIterator {
    // Important that this starts at -1. Need a boolean to denote that it has started iterating.
    private int docID = -1;
    private boolean started = false;
    private double curScore = 0;
    private final HitCounter counter;
    private final KthGreatest.Result kgr;
    private final MatchHashesAndScoreQuery.ScoreFunction scoreFunction;
    private final double maxScore;
    private final int candidates;
    // Track the number of ids emitted, and the number of ids with count = kgr.kthGreatest emitted.
    private int numEmitted = 0;
    private int numEq = 0;

    public MaxScoreDocIdSetIterator(HitCounter hitCounter, KthGreatest.Result kgr, MatchHashesAndScoreQuery.ScoreFunction scoreFunction, int candidates,double maxScore) {
        this.counter = hitCounter;
        this.kgr = kgr;
        this.scoreFunction = scoreFunction;
        this.candidates = candidates;
        this.maxScore = maxScore;
    }


    public double curScore() {
        return curScore;
    }

    @Override
    public int docID() {
        return docID;
    }

    @Override
    public int nextDoc() {

        if (!started) {
            started = true;
            docID = counter.minKey() - 1;
        }

        // Ensure that docs with count = kgr.kthGreatest are only emitted when there are fewer
        // than `candidates` docs with count > kgr.kthGreatest.
        while (true) {
            if (numEmitted == candidates || docID + 1 > counter.maxKey()) {
                docID = DocIdSetIterator.NO_MORE_DOCS;
                curScore = 0;
                return docID();
            } else {
                docID++;
                if (counter.get(docID) > kgr.kthGreatest) {
                    double score = scoreFunction.score(docID(), counter.get(docID()));
                    if (score > 4) {
                        numEmitted++;
                        curScore = score;
                        return docID();
                    }

                } else if (counter.get(docID) == kgr.kthGreatest && numEq < candidates - kgr.numGreaterThan) {
                    double score = scoreFunction.score(docID(), counter.get(docID()));
                    if (score > 4) {
                        numEq++;
                        numEmitted++;
                        curScore = score;
                        return docID();
                    }
                }
            }
        }
    }

    @Override
    public int advance(int target) {
        while (docID < target) nextDoc();
        return docID();
    }

    @Override
    public long cost() {
        return counter.numHits();
    }


}