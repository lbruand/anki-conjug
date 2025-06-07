import random
import re
import unittest
import json
from dataclasses import dataclass
from pathlib import Path

import genanki
import ollama
import spacy
import tqdm
from pydantic import BaseModel
from verbecc import Conjugator


def printjson(c):
    print(json.dumps(c, indent=4, ensure_ascii=False))
# verbs
verbs = ["ser", "tener", "hacer", "ir", "decir", "salir", "saber",
         "vivir", "comer", "hablar", "conducir", "ver", "mirar",
         "poner", "poder", "pensar", "sentir", "dormir", "leer",
         "abrir", "conocer", "traer", "averiguar", "incontrar",
         "acabar"]

# Spanish subject pronouns
pronouns = ["yo", "tú", "él", "nosotros", "vosotros", "ellos"]
pronouns_imperativo = pronouns[1:]


@dataclass(frozen=True, unsafe_hash=True)
class Note:
    question: str
    response: str


class ResultSafe(BaseModel):
  safe: bool
  grammar: bool
  spanish: bool


def generate(cg = Conjugator(lang='es'),
             rand = random.Random(3),
             nlp = spacy.load("es_core_news_sm")):

    mood = "imperativo"
    verb_infinitive = rand.choice(verbs)
    conjug = cg.conjugate(verb_infinitive)
    #verb_infinitive = conjug['verb']['inifinitive']
    conjug_imperativo = conjug["moods"]["imperativo"]
    tipo = rand.choice(list(conjug_imperativo.values()))
    alpha, beta = 0.3, 0.1

    pronoun, conjug_verb = rand.choices(population=list(zip(pronouns_imperativo, tipo)),
                                        weights=[alpha, beta, beta, alpha, beta], k=1)[0]
    print(f"{pronoun} {conjug_verb}")
    result_content = None
    recursion_guard = 5
    while not(result_content) and recursion_guard>0:
        recursion_guard -= 1
        result_content = generate_from_model(conjug_verb, rand, tipo)
        if not check_conjug_verb_presence(conjug_verb, result_content, nlp):
            print(f"iter=[{recursion_guard}] => [{conjug_verb}] is not in [{result_content}]")
            result_content = None
            continue
        check_response_content = guardrail(rand, result_content)

        if not check_response_content.safe or not check_response_content.spanish or not check_response_content.grammar:
            print(f"check_response[{recursion_guard}] = [{check_response_content}]")
            result_content = None
            continue


    print(f"generated sentence = [{result_content}]")
    if result_content is None:
        return None
    replace_value = f"___ ({pronoun} {verb_infinitive})"
    result_content_ablated = re.sub(conjug_verb, replace_value, result_content, flags=re.IGNORECASE)
    print(result_content_ablated)
    replace_value_highlight = r"<font color='red'>\1</font>"
    result_content_highlighted = re.sub(f"({conjug_verb})",
                                        replace_value_highlight,
                                        result_content, flags=re.IGNORECASE)
    return dict(question=result_content_ablated,
                response=result_content_highlighted)


def guardrail(rand, result_content):
    check_response = ollama.chat(model="mistral",
                                 messages=[{"role": "user",
                                            "content":
                                                f"""Spellcheck and control the grammar of the spanish sentence:
[{result_content}]
Just return 'safe' if the grammar is correct and it is simple spanish. 
return only JSON:
        """}, ],
                                 stream=False,
                                 format=ResultSafe.model_json_schema(),
                                 options={
                                     'seed': rand.randint(0, 2 << 15)
                                     # "temperature": 0
                                 }
                                 )

    return ResultSafe.model_validate_json(check_response.message.content)


def check_conjug_verb_presence(conjug_verb, result_content, nlp = spacy.load("es_core_news_sm")):
    conjug_verb_tokens = list(str(token).lower() for token in nlp(conjug_verb))
    result_content_tokens = list(str(token).lower() for token in nlp(result_content))
    return ispresent_sublist_in_list(conjug_verb_tokens, result_content_tokens)


def generate_from_model(conjug_verb, rand, tipo):
    response = ollama.chat(model="llama3.2",
                           messages=[{"role": "user",
                                      "content":
                                          f"""You are a spanish teacher. Write a simple spanish sentence using the
                               verb '{conjug_verb}' in the imperativo {tipo} form. 
                               do not translate it. Write only that sentence in spanish.
"""}],
                           stream=False,
                           options={
                               'seed': rand.randint(0, 2 << 15)
                               # "temperature": 0
                           }
                           )
    return response.message.content

def ispresent_sublist_in_list(sublist, l):
    if len(l) < len(sublist):
        return False
    else:
        if sublist == l[:len(sublist)]:
            return True
        else:
            return ispresent_sublist_in_list(sublist, l[1:])


class MyTestCase(unittest.TestCase):
    def test_find_sublist_in_list(self):
        self.assertTrue(ispresent_sublist_in_list(list(range(0, 3)), list(range(0, 8))))
        self.assertTrue(ispresent_sublist_in_list(list(range(3, 5)), list(range(0, 8))))
        self.assertFalse(ispresent_sublist_in_list(list(range(7, 11)), list(range(0, 8))))

    def test_conjub_verb_presence(self):
        self.assertFalse(check_conjug_verb_presence("coma", "Coman todos."))
        self.assertTrue(check_conjug_verb_presence("vivamos", "Vivamos al aire libre todos los fines de semana."))
        self.assertTrue(check_conjug_verb_presence("no averigüéis",
                                                   "No averigüéis quién lo hizo."))

    def test_generate(self):
        print(generate())

    def test_wholedeck(self):
        rand = random.Random(3)

        work = [generate(cg=Conjugator(lang='es'), rand=rand, nlp=spacy.load("es_core_news_sm")) for i in range(50)]
        notes = set([item for item in work if item])

        output_dir = Path('output')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'imperative.apkg'

        my_model = genanki.Model(
            16555892359,
            'Simple Model',
            fields=[
                {'name': 'Question'},
                {'name': 'Answer'},
            ],
            templates=[
                {
                    'name': 'Card 1',
                    'qfmt': '<div class="box"><div class="box-inside">{{Question}}</div></div>',
                    'afmt': '<div class="box"><div class="box-inside">{{Answer}}</div></div>',
                },
            ],
            css="""
.card {
  font-size: 20px;
  text-align: center;
  color: black;
  background-image: url("_background_tiles2.jpg");
}
.box {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    box-sizing: border-box;
    display: inline-block;
    min-width: 92%;
    min-height:35%;
    background-color: rgba(255, 255, 255, 0.80);
    border-radius: 10% 30% 50% 70%;
}
.box-inside {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    box-sizing: border-box;
    display: inline-block;
    min-width: 92%;
    color: black;
}
.nightMode .box {
    background-color: rgba(0, 0, 0, 0.80);
}
            """
        )
        my_deck = genanki.Deck(
            205940084710,
            'Imperativo')
        for note in tqdm.tqdm(notes):
            my_note = genanki.Note(
                model=my_model,
                fields=[note.question,
                        note.response])
            my_deck.add_note(my_note)

        my_package = genanki.Package(my_deck)
        my_package.media_files = ['../img/_background_tiles2.jpg']
        my_package.write_to_file(output_file)
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
