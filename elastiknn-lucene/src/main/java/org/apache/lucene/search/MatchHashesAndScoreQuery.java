package org.apache.lucene.search;

import com.klibisz.elastiknn.search.ArrayHitCounter;
import com.klibisz.elastiknn.search.HitCounter;
import com.klibisz.elastiknn.models.HashAndFreq;
import org.apache.lucene.index.*;
import org.apache.lucene.util.BytesRef;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;

import static java.lang.Math.max;
import static java.lang.Math.min;

/**
 * Query that finds docs containing the given hashes hashes (Lucene terms), and then applies a scoring function to the
 * docs containing the most matching hashes. Largely based on Lucene's TermsInSetQuery.
 */
public class MatchHashesAndScoreQuery extends Query {

    public interface ScoreFunction {
        double score(int docId, int numMatchingHashes);
    }

    private final String field;
    private final HashAndFreq[] hashAndFrequencies;
    private final int candidates;
    private final IndexReader indexReader;
    private final Function<LeafReaderContext, ScoreFunction> scoreFunctionBuilder;
    private final Logger logger;
    private final double minScore;

    public MatchHashesAndScoreQuery(final String field,
                                    final HashAndFreq[] hashAndFrequencies,
                                    final int candidates,
                                    final IndexReader indexReader,
                                    final Function<LeafReaderContext, ScoreFunction> scoreFunctionBuilder,
                                    final double minScore) {
        // `countMatches` expects hashes to be in sorted order.
        // java's sort seems to be faster than lucene's ArrayUtil.
        java.util.Arrays.sort(hashAndFrequencies, HashAndFreq::compareTo);

        this.field = field;
        this.hashAndFrequencies = hashAndFrequencies;
        this.candidates = candidates;
        this.indexReader = indexReader;
        this.scoreFunctionBuilder = scoreFunctionBuilder;
        this.logger = LogManager.getLogger(getClass().getName());
        this.minScore = minScore;
    }

    public MatchHashesAndScoreQuery(final String field,
                                    final HashAndFreq[] hashAndFrequencies,
                                    final int candidates,
                                    final IndexReader indexReader,
                                    final Function<LeafReaderContext, ScoreFunction> scoreFunctionBuilder
                                    ) {
        this(field, hashAndFrequencies, candidates, indexReader, scoreFunctionBuilder, Double.NEGATIVE_INFINITY);
    }

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) {

        return new Weight(this) {

            /**
             * Builds and returns a map from doc ID to the number of matching hashes in that doc.
             */
            private HitCounter countHits(LeafReader reader) throws IOException {
                Terms terms = reader.terms(field);
                // terms seem to be null after deleting docs. https://github.com/alexklibisz/elastiknn/issues/158
                if (terms == null) {
                    return new ArrayHitCounter(0);
                } else {
                    TermsEnum termsEnum = terms.iterator();
                    PostingsEnum docs = null;
                    HitCounter counter = new ArrayHitCounter(reader.maxDoc());
                    double counterLimit = counter.capacity() + 1;
                    // TODO: Is this the right place to use the live docs bitset to check for deleted docs?
                    // Bits liveDocs = reader.getLiveDocs();
                    for (HashAndFreq hf : hashAndFrequencies) {
                        if (counter.numHits() < counterLimit && termsEnum.seekExact(new BytesRef(hf.hash))) {
                            docs = termsEnum.postings(docs, PostingsEnum.NONE);
                            while (docs.nextDoc() != DocIdSetIterator.NO_MORE_DOCS && counter.numHits() < counterLimit) {
                                    counter.increment(docs.docID(), min(hf.freq, docs.freq()));
                            }
                        }
                    }
                    return counter;
                }
            }

            private DocIdSetIterator buildDocIdSetIterator(HitCounter counter,ScoreFunction scoreFunction) {
                if (counter.numHits() < candidates) {
                    logger.warn(String.format(
                            "Found fewer approximate matches [%d] than the requested number of candidates [%d]",
                            counter.numHits(), candidates));
                }
                if (counter.isEmpty()) return DocIdSetIterator.empty();
                else {

                    KthGreatest.Result kgr = counter.kthGreatest(candidates);

                    // Return an iterator over the doc ids >= the min candidate count.
                    return new MaxScoreDocIdSetIterator(counter,kgr,scoreFunction,candidates, minScore);
                }
            }

            @Override
            public void extractTerms(Set<Term> terms) { }

            @Override
            public Explanation explain(LeafReaderContext context, int doc) throws IOException {
                HitCounter counter = countHits(context.reader());
                if (counter.get(doc) > 0) {
                    ScoreFunction scoreFunction = scoreFunctionBuilder.apply(context);
                    double score = scoreFunction.score(doc, counter.get(doc));
                    return Explanation.match(score, String.format("Document [%d] and the query vector share [%d] of [%d] hashes. Their exact similarity score is [%f].", doc, counter.get(doc), hashAndFrequencies.length, score));
                } else {
                    return Explanation.noMatch(String.format("Document [%d] and the query vector share no common hashes.", doc));
                }
            }

            @Override
            public Scorer scorer(LeafReaderContext context) throws IOException {
                ScoreFunction scoreFunction = scoreFunctionBuilder.apply(context);
                LeafReader reader = context.reader();
                HitCounter counter = countHits(reader);
                DocIdSetIterator disi =  buildDocIdSetIterator(counter,scoreFunction);

                return new Scorer(this) {
                    @Override
                    public DocIdSetIterator iterator() {
                        return disi;
                    }

                    @Override
                    public float getMaxScore(int upTo) {
                        return Float.MAX_VALUE;
                    }

                    @Override
                    public float score() {
                        int docID = docID();

                        // TODO: how does it get to this state? This error did come up once in some local testing.
                        if (docID == DocIdSetIterator.NO_MORE_DOCS) return 0f;
                        else return (float) ((MaxScoreDocIdSetIterator) disi).curScore();
                    }

                    @Override
                    public int docID() {
                        return disi.docID();
                    }
                };
            }

            @Override
            public boolean isCacheable(LeafReaderContext ctx) {
                return false;
            }
        };
    }

    @Override
    public String toString(String field) {
        return String.format(
                "%s for field [%s] with [%d] hashes and [%d] candidates",
                this.getClass().getSimpleName(),
                this.field,
                this.hashAndFrequencies.length,
                this.candidates);
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof MatchHashesAndScoreQuery) {
            MatchHashesAndScoreQuery q = (MatchHashesAndScoreQuery) obj;
            return q.hashCode() == this.hashCode();
        } else {
            return false;
        }
    }

    @Override
    public int hashCode() {
        return Objects.hash(field, hashAndFrequencies, candidates, indexReader, scoreFunctionBuilder);
    }
}
