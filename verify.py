import nltk

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

#first = ['A', 'group', 'of', 'people', 'in', 'blue', 'shirts', 'at', 'a', 'sporting', 'event', '.']
#second = ['A', 'group', 'of', 'people', 'in', 'blue', 'shirts', 'are', 'at', 'a', 'sporting', 'event', '.']
first = ["thank", "you", "."]
second = ["thank", "you", "."]

#first = [" ".join(first)]
#second = " ".join(second)
first = [first]

print(first)
print(second)

print(sentence_bleu(first, second, smoothing_function=SmoothingFunction().method1))
print(nltk.translate.bleu_score.corpus_bleu(first, second, smoothing_function=SmoothingFunction().method1))
