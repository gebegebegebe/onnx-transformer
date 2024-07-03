import nltk

from nltk.translate.bleu_score import sentence_bleu

first = ['A', 'group', 'of', 'people', 'in', 'blue', 'shirts', 'at', 'a', 'sporting', 'event', '.']
second = ['A', 'group', 'of', 'people', 'in', 'blue', 'shirts', 'are', 'at', 'a', 'sporting', 'event', '.']

#first = [" ".join(first)]
#second = " ".join(second)
first = [first]

print(first)
print(second)

print(sentence_bleu(first, second))
