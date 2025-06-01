import random
import unittest
import json
from dataclasses import dataclass
from pathlib import Path

import genanki
import tqdm
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

# Triggers
conjunction_subjunctive_triggers = [
    "Es importante que", "Quiero que", "Es posible que", "Dudo que", "Espero que",
    "Me alegra que", "Ojalá que", "Temo que", "Es necesario que", "No creo que",
    "Es increíble que"
]
conjunction_indicative_triggers = [
    "Es cierto que", "Sé que", "Creo que", "Es verdad que", "Está claro que",
    "Me parece que", "No dudo que", "Es obvio que", "Está demostrado que", "Afirmo que"
]

# Relative clause triggers
relative_indicative_triggers = [
    "Tengo un amigo que", "Conozco a alguien que", "Hay un profesor que",
    "Veo una persona que", "Ella tiene un perro que"
]

relative_subjunctive_triggers = [
    "Busco un amigo que", "Quiero una persona que", "Necesito un profesor que",
    "No hay nadie que", "¿Conoces a alguien que"
]

@dataclass(frozen=True, unsafe_hash=True)
class Note:
    question: str
    response: str

def generate(cg = Conjugator(lang='es'), rand = random.Random(3)):
    tipo = rand.choice(["relative", "conjunction"])
    mood = rand.choices(population=["indicativo", "subjuntivo"], weights=[0.3, 0.7], k=1)[0]
    verb = rand.choice(verbs)
    conjug = cg.conjugate(verb)
    if mood in conjug['moods']:
        verb_conjug = conjug['moods'][mood]['presente']
    else:
        print(verb)
        print(conjug['moods'].keys())
    if tipo == "conjunction":
        triggers = conjunction_subjunctive_triggers if mood == "subjuntivo" else conjunction_indicative_triggers
        verb_form = rand.choice(verb_conjug)
        subject_nospace, conjug_verb = verb_form.split(' ')
        subject = f'{subject_nospace} '
    else:
        triggers = relative_subjunctive_triggers if mood == "subjuntivo" else relative_indicative_triggers
        verb_form = verb_conjug[2]
        _, conjug_verb = verb_form.split(' ')
        subject = ''
    trigger = rand.choice(triggers)
    return Note(question=f"{trigger} {subject}___ ({verb})",
                response=f"{trigger} {subject}<font color='red'>{conjug_verb}</font> ({mood})")
    #return {"question": f"{trigger} {subject}___ ({verb})",
    #       "response": f"{trigger} {subject}<font color='red'>{conjug_verb}</font> ({mood})"}


class MyTestCase(unittest.TestCase):
    def test_something(self):
        rand = random.Random(3)

        notes = set([generate(rand=rand) for i in range(250)])

        output_dir = Path('output')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'subjuntivo.apkg'

        my_model = genanki.Model(
            16555892319,
            'Simple Model',
            fields=[
                {'name': 'Question'},
                {'name': 'Answer'},
            ],
            templates=[
                {
                    'name': 'Card 1',
                    'qfmt': '{{Question}}',
                    'afmt': '{{Answer}}',
                },
            ])
        my_deck = genanki.Deck(
            205940084710,
            'Subjuntivo')
        for note in tqdm.tqdm(notes):
            my_note = genanki.Note(
                model=my_model,
                fields=[note.question,
                        note.response])
            my_deck.add_note(my_note)
        #sentences = [
        #    ("Busco una persona que ___ (saber) japonés.", "subjunctivo"),
        #    ("Tengo un amigo que ___ (vivir) en Chile. ", "indicativo"),
        #    ("¿Conoces a alguien que ___ (tocar) el violín?", "subjunctivo"),
        #    ("Hay un chico aquí que ______ (hablar) italiano.", "indicativo"),
        #]
        #print(conjug['moods'].keys()) # ['subjuntivo'])
        genanki.Package(my_deck).write_to_file(output_file)
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
