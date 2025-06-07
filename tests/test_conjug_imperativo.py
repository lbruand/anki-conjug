import io
import random
import re
import unittest
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Set
from wave import Wave_read

import genanki
import mp3
import numpy as np
import ollama
import scipy
import spacy
import tqdm
from pydantic import BaseModel
from transformers import AutoProcessor, BarkModel
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


@dataclass
class Note:
    question: str
    response: str
    original_sentence: str
    reading: Path = None
    def __hash__(self):
        return hash(self.original_sentence)

    def __eq__(self, other):
        return self.original_sentence.__eq__(other.original_sentence)

class ResultSafe(BaseModel):
  safe: bool
  grammar: bool
  spanish: bool


def generate(cg = Conjugator(lang='es'),
             rand = random.Random(3),
             nlp = spacy.load("es_core_news_sm")) -> Note:

    mood = "imperativo"
    verb_infinitive = rand.choice(verbs)
    conjug = cg.conjugate(verb_infinitive)
    #verb_infinitive = conjug['verb']['inifinitive']
    conjug_imperativo = conjug["moods"][mood]
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
    return Note(question=result_content_ablated,
                response=result_content_highlighted,
                original_sentence=result_content)


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


def generate_audio(sentence: str,
                   output_file: Path,
                   voice_preset = "v2/es_speaker_8",
                   processor = AutoProcessor.from_pretrained("suno/bark"),
                   model = BarkModel.from_pretrained("suno/bark"),
                   ) -> np.array:
    inputs = processor(f"- {sentence}", voice_preset=voice_preset)

    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()
    sample_rate = model.generation_config.sample_rate

    max_val = np.max(np.abs(audio_array))

    if max_val > 0:
        data_normalized = audio_array / max_val
    else:
        data_normalized = audio_array

    data_int16 = np.int16(data_normalized * 32767)

    with io.BytesIO() as fout:
        scipy.io.wavfile.write(fout, rate=sample_rate, data=data_int16)
        all_bytes = fout.getvalue()

    with io.BytesIO(all_bytes) as fin:
        wav_file = Wave_read(fin)

        sample_size = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        nchannels = wav_file.getnchannels()

        if sample_size != 2:
            raise ValueError("Only PCM 16-bit sample size is supported (input audio: %s)" % sample_size)

        with open(output_file, 'wb') as fout:
            encoder = mp3.Encoder(fout)
            encoder.set_bit_rate(64)
            encoder.set_sample_rate(sample_rate)
            encoder.set_channels(nchannels)
            encoder.set_quality(5)
            encoder.set_mode(mp3.MODE_SINGLE_CHANNEL)
            while True:
                pcm_data = wav_file.readframes(8000)
                if pcm_data:
                    encoder.write(pcm_data)
                else:
                    encoder.flush()
                    break

    return audio_array

class MyTestCase(unittest.TestCase):

    def test_generate_speech(self):
        print("hello")
        output_dir = Path('output')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'vivamos.mp3'
        audio_array = generate_audio("Vivamos al aire libre todos los fines de semana.",
                                     output_file)


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

        cg = Conjugator(lang='es')
        nlp = spacy.load("es_core_news_sm")
        work = [generate(cg=cg, rand=rand, nlp=nlp) for i in range(70)]
        notes: Set[Note] = set([item for item in work if item])

        output_dir = Path('output')
        output_dir.mkdir(parents=True, exist_ok=True)
        media_files = ['../img/_background_tiles2.jpg']

        voice_preset = "v2/es_speaker_8"
        processor = AutoProcessor.from_pretrained("suno/bark")
        model = BarkModel.from_pretrained("suno/bark")
        for ix, card in tqdm.tqdm(enumerate(notes)):
            output_file = output_dir / f"card_{ix}.mp3"
            generate_audio(card.original_sentence,
                           output_file,
                           voice_preset,
                           processor,
                           model)
            card.reading = f"[sound:{output_file.name}]"
            media_files.append(output_file)

        output_file = output_dir / 'imperative.apkg'

        my_model = genanki.Model(
            16555892359,
            'Simple Model',
            fields=[
                {'name': 'Question'},
                {'name': 'Answer'},
                {'name': 'Reading'},
            ],
            templates=[
                {
                    'name': 'Card 1',
                    'qfmt': '<div class="box"><div class="box-inside">{{Question}}</div></div>',
                    'afmt': '<div class="box"><div class="box-inside">{{Answer}}<br/>{{Reading}}</div></div>',
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
            205940088710,
            'Imperativo')
        for note in tqdm.tqdm(notes):
            my_note = genanki.Note(
                model=my_model,
                fields=[note.question,
                        note.response,
                        note.reading])
            my_deck.add_note(my_note)

        my_package = genanki.Package(my_deck)

        my_package.media_files = media_files
        my_package.write_to_file(output_file)
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
