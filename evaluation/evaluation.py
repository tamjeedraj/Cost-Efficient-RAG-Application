import json
import numpy as np

from app.rag import retrieve


def recall_at_k(retrieved, relevant):

    return 1 if relevant in retrieved else 0


def hit_rate(retrieved, relevant):

    return 1 if relevant in retrieved else 0


def reciprocal_rank(retrieved, relevant):

    for i, item in enumerate(retrieved):

        if item == relevant:

            return 1 / (i + 1)

    return 0


def evaluate():

    with open("evaluation/questions.json") as f:

        dataset = json.load(f)

    recalls = []

    hits = []

    mrr = []

    for sample in dataset:

        result = retrieve(sample["question"])

        retrieved = []

        for meta in result["metadatas"][0]:

            retrieved.append(meta["chunk"])

        recalls.append(

            recall_at_k(
                retrieved,
                sample["relevant_chunk"]
            )
        )

        hits.append(

            hit_rate(
                retrieved,
                sample["relevant_chunk"]
            )
        )

        mrr.append(

            reciprocal_rank(
                retrieved,
                sample["relevant_chunk"]
            )
        )

    print("Recall@k :", np.mean(recalls))

    print("Hit Rate :", np.mean(hits))

    print("MRR :", np.mean(mrr))


if __name__ == "__main__":

    evaluate()