# coding: utf-8

from __future__ import division
from nltk.tokenize import word_tokenize

import models
import data

import theano
import tornado.ioloop
import tornado.web
import re

import theano.tensor as T
import numpy as np

from tornado.options import define, options

define('port', default=8080, help='run on the given port', type=int)

### CONFIGURATION ###

MAX_TOTAL_LENGTH = 50000

LANGUAGES = {
    'EN': {
        'model_path': './Demo-Europarl-EN.pcl',
        'untokenizer': lambda text: text.replace(" '", "'").replace(" n't", "n't").replace("can not", "cannot"),
        },
}

### END OF CONFIGURATION ###

default_tokenizer = word_tokenize
default_untokenizer = lambda text: text

class Punctuator(object):

    def __init__(self, conf):
        self.model_path = conf['model_path']
        self.tokenizer = conf.get('tokenizer', default_tokenizer)
        self.untokenizer = conf.get('untokenizer', default_untokenizer)
        self.lowercase = conf.get('lowercase', True)

        x = T.imatrix('x')
        net, _ = models.load(self.model_path, 1, x)

        self.predict=theano.function(inputs=[x], outputs=net.y)
        self.word_vocabulary=net.x_vocabulary
        self.punctuation_vocabulary=net.y_vocabulary
        self.reverse_punctuation_vocabulary = {v:k for k,v in self.punctuation_vocabulary.items()}
        self.human_readable_punctuation_vocabulary = [p[0] for p in self.punctuation_vocabulary if p != data.SPACE]

numbers = re.compile(r'\d')
is_number = lambda x: len(numbers.sub('', x)) / len(x) < 0.6

def convert_punctuation_to_readable(punct_token):
    if punct_token == data.SPACE:
        return ' '
    elif punct_token.startswith('-'):
        return ' ' + punct_token[0] + ' '
    else:
        return punct_token[0] + ' '

def punctuate(words, word_vocabulary, predict, reverse_punctuation_vocabulary, lowercase, writer=None):
    if len(words) == 0:
        return

    if words[-1] != data.END:
        words += [data.END]

    i = 0

    while True:

        subsequence = words[i:i+data.MAX_SEQUENCE_LEN]

        if len(subsequence) == 0:
            break

        converted_subsequence = [word_vocabulary.get(data.NUM
            if is_number(w)
            else (w.lower() if lowercase else w),
            word_vocabulary[data.UNK]) for w in subsequence]

        y = predict(np.array([converted_subsequence], dtype=np.int32).T)

        if writer:
            writer.write(subsequence[0].title())

        last_eos_idx = 0
        punctuations = []
        for y_t in y:

            p_i = np.argmax(y_t.flatten())
            punctuation = reverse_punctuation_vocabulary[p_i]

            punctuations.append(punctuation)

            if punctuation in data.EOS_TOKENS:
                last_eos_idx = len(punctuations) # we intentionally want the index of next element

        if subsequence[-1] == data.END:
            step = len(subsequence) - 1
        elif last_eos_idx != 0:
            step = last_eos_idx
        else:
            step = len(subsequence) - 1

        for j in range(step):
            current_punctuation = punctuations[j]
            yield current_punctuation
            if writer:
                writer.write(convert_punctuation_to_readable(current_punctuation))
                if j < step - 1:
                    if current_punctuation in data.EOS_TOKENS:
                        writer.write(subsequence[1+j].title())
                    else:
                        writer.write(subsequence[1+j])

        if writer:
            writer.flush()

        if subsequence[-1] == data.END:
            break

        i += step

        if i > MAX_TOTAL_LENGTH:
            break

### HANDLERS ###

class BaseHandler(tornado.web.RequestHandler):
    
    def initialize(self, punctuators):
        self.punctuators = punctuators

    def get_punctuator(self, language):
        if not language:
            language = 'EN'

        if language not in self.punctuators:
            raise tornado.web.HTTPError(404)

        return punctuators[language]        

class MainHandler(BaseHandler):

    def post(self, language):
        punctuator = self.get_punctuator(language)

        text = self.get_argument('text', '')
        words = [w for w in punctuator.untokenizer(' '.join(punctuator.tokenizer(text))).split()
                 if w not in punctuator.punctuation_vocabulary and w not in punctuator.human_readable_punctuation_vocabulary]

        list(punctuate(words, punctuator.word_vocabulary, punctuator.predict, punctuator.reverse_punctuation_vocabulary, punctuator.lowercase, writer=self))

if __name__ == '__main__':

    tornado.options.parse_command_line()

    punctuators = {}
    for language, conf in LANGUAGES.items():
        print('Initializing %s...' % language)
        punctuators[language] = Punctuator(conf)

    init_params = dict(punctuators=punctuators)

    print('Serving...')
    application = tornado.web.Application([
        (r'/?(?P<language>[a-zA-Z]+)?/?', MainHandler, init_params),
    ])
    application.listen(options.port)
    tornado.ioloop.IOLoop.current().start()