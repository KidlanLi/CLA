import sys

from hw05_word_similarity.word_similarity import PpmiWeightedSparseMatrix
from nltk.corpus import brown

def main(argv):
    print('Loading data ...')
    word_list = list(brown.words())
    sim_matrix = PpmiWeightedSparseMatrix(word_list, vocab_size=10000, window_size=1)
    print("Computing singular value decomposition ...")
    sim_matrix_svd = sim_matrix.toSvdSimilarityMatrix(n_components=50)
    print("... done!")

    while True:
        query_word = input("Word: ").strip()
        if not query_word:
            break
        print("Most similar (no SVD): ")
        for w in sim_matrix.most_similar_words(query_word, 5):
            print(w)
        print("Most similar (with SVD): ")
        for w in sim_matrix_svd.most_similar_words(query_word, 5):
            print(w)
        print()

if __name__ == "__main__":
    main(sys.argv[1:])
