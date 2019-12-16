from custom.analysis import Analysis, Doc2VecAnalysis
from custom.utils import load_model


class Logger:
    def __init__(self, model_name, type, model=None):
        self.log_path = '../log/'
        self.model_path = '../models/'
        self.model_name = model_name
        self.type = type
        self.model = model


class AnalysisLogger(Logger):
    def init_model(self):
        if self.type == "d2v":
            self.model = load_model(self.model_path, self.model_name, self.type)

    def write_stats_to_file(self):
        self.init_model()
        if self.type == "d2v":
            analysis = Doc2VecAnalysis(model_name=self.model_name, model=self.model, type='d2v')
        else:
            analysis = Analysis(model_name=self.model_name, model=self.model, type='w2v')

        with open(self.log_path + "analysis/logger_" + self.model_name + ".txt") as fp:
            title = "Model: " + self.model_name
            print("\n" + title + "\n" + ("-" * (len(title) + 1)))
            fp.write('Vocab Size:', analysis.get_vocab_size(), end="\n")
            fp.write('Vocab Size:', analysis.get_vocab_size(), end="\n")
            fp.write('Document Count:', analysis.get_number_of_docs(), end="\n")
            fp.write('Apple:', analysis.most_similar_words('apple'), end="\n")
            fp.write('Google:', analysis.most_similar_words('google'), end="\n")
            fp.write('Tesla:', analysis.most_similar_words('tesla'), end="\n")
            fp.write('Microsoft:', analysis.most_similar_words('microsft'), end="\n")
            fp.write('King + Woman - Man =', analysis.subtract_from_vectors('king', 'woman', 'man'), end="\n")
            fp.write('{0} + {1} - {2} ='.format('Paris', 'England', 'London'),
                     analysis.subtract_from_vectors('paris', 'england', 'london'))
            fp.write('{0} + {1} - {2} ='.format('Microsoft', 'Google', 'Windows'),
                     analysis.subtract_from_vectors('microsoft', 'google', 'windows'))
