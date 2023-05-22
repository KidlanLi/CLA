import sys

from hw06_skipgram import skipgram, utils
from nltk.corpus import brown

def main(argv):
    skipgram_model = skipgram.SkipGram(brown.words())
    print("Taining skipgram embeddings ...")
    iters = 2
    for i in range(iters):
        print("Iteration %d out of %d" % (i+1, iters))
        skipgram_model.train_iter()
    print("... done!")
    skipgram_similarity = skipgram_model.toDenseSimilarityMatrix()

    while True:
        query_word = input("Word: ").strip()
        if not query_word:
            break
        print("Most similar: ")
        for w in skipgram_similarity.most_similar_words(query_word, 5):
            print(w)
        print()

if __name__ == "__main__":
    main(sys.argv[1:])
