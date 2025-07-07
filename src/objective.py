import logging
import re
from typing import Tuple

import nltk
import numpy as np
from nltk.corpus import wordnet as wn


class ObjectiveTest:
	"""Class abstraction for objective test generation module."""

	def __init__(self, filepath: str):
		"""Class constructor.

		Args:
			filepath (str): Absolute filepath to the subject corpus.
		"""
		# Load subject corpus
		try:
			with open(filepath, mode="r") as fp:
				self.summary = fp.read()
		except FileNotFoundError:
			logging.exception("Corpus file not found.", exc_info=True)
			self.summary = ""  # Ensure it doesn't break later

	def generate_test(self, num_questions: int = 3) -> Tuple[list, list]:
		"""Method to generate an objective test.

		Args:
			num_questions (int, optional): Number of questions in a test. Defaults to 3.

		Returns:
			Tuple[list, list]: Questions and answer options respectively.
		"""
		question_sets = self.get_question_sets()

		question_answers = [
			qs for qs in question_sets if qs["Key"] > 3
		]

		questions, answers = [], []
		while len(questions) < min(num_questions, len(question_answers)):
			rand_num = np.random.randint(0, len(question_answers))
			q = question_answers[rand_num]
			if q["Question"] not in questions:
				questions.append(q["Question"])
				answers.append(q["Answer"])

		return questions, answers

	def get_question_sets(self) -> list:
		"""Identify sentences with potential objective questions.

		Returns:
			list: Sentences with potential objective questions.
		"""
		try:
			sentences = nltk.sent_tokenize(self.summary)
		except Exception:
			logging.exception("Sentence tokenization failed.", exc_info=True)
			return []

		question_sets = []
		for sent in sentences:
			qs = self.identify_potential_questions(sent)
			if qs is not None:
				question_sets.append(qs)
		return question_sets

	def identify_potential_questions(self, sentence: str) -> dict:
		"""Identify potential question sets.

		Args:
			sentence (str): Tokenized sequence from corpus.

		Returns:
			dict: Question formed along with the correct answer or None.
		"""
		# POS tagging
		try:
			tokens = nltk.word_tokenize(sentence)
			tags = nltk.pos_tag(tokens)
			if tags[0][1] == "RB" or len(tokens) < 4:
				return None
		except Exception:
			logging.exception("POS tagging failed.", exc_info=True)
			return None

		# Chunking grammar
		noun_phrases = []
		grammar = r"""
			CHUNK: {<NN>+<IN|DT>*<NN>+}
			       {<NN>+<IN|DT>*<NNP>+}
			       {<NNP>+<NNS>*}
		"""

		chunker = nltk.RegexpParser(grammar)
		tree = chunker.parse(tags)

		for subtree in tree.subtrees():
			if subtree.label() == "CHUNK":
				phrase = " ".join(word for word, _ in subtree).strip()
				noun_phrases.append(phrase)

		# Replace nouns with blanks
		replace_nouns = []
		for word, _ in tags:
			for phrase in noun_phrases:
				if phrase and phrase[0] == '\'':
					break
				if word in phrase:
					replace_nouns.extend(phrase.split()[-2:])
					break
			if not replace_nouns:
				replace_nouns.append(word)
			break

		if not replace_nouns:
			return None

		val = min(len(w) for w in replace_nouns)

		trivial = {
			"Answer": " ".join(replace_nouns),
			"Key": val
		}

		if len(replace_nouns) == 1:
			trivial["Similar"] = self.answer_options(replace_nouns[0])
		else:
			trivial["Similar"] = []

		replace_phrase = " ".join(replace_nouns)
		blanks_phrase = ("__________ " * len(replace_nouns)).strip()
		expression = re.compile(re.escape(replace_phrase), re.IGNORECASE)
		trivial["Question"] = expression.sub(blanks_phrase, sentence, count=1)

		return trivial

	@staticmethod
	def answer_options(word: str) -> list:
		"""Generate incorrect answer options using WordNet.

		Args:
			word (str): Actual correct answer.

		Returns:
			list: List of incorrect options.
		"""
		try:
			synsets = wn.synsets(word, pos="n")
		except Exception:
			logging.exception("Synsets creation failed.", exc_info=True)
			return []

		if not synsets:
			return []

		synset = synsets[0]
		hypernyms = synset.hypernyms()

		if not hypernyms:
			return []

		hyponyms = hypernyms[0].hyponyms()
		similar_words = []

		for hyponym in hyponyms:
			lemma = hyponym.lemmas()[0].name().replace("_", " ")
			if lemma != word:
				similar_words.append(lemma)
			if len(similar_words) == 8:
				break

		return similar_words
