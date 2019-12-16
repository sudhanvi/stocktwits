from datetime import datetime

from custom.utils import load_model


class Analysis:
    def __init__(self, model_name, model, type):
        self.path_data = '../data/'
        self.path_models = '../models/'
        self.path_logs = '../log/'
        self.type = type
        self.model_name = model_name
        if self.type == "d2v":
            self.model = load_model(self.path_models + self.model_name + "/", self.model_name + ".model", self.type)
            self.w2v = self.model.wv
        else:
            self.model = model
            self.w2v = self.model

    def infer_vector(self, test_data_list):
        # to find the vector of a document which is not in training data
        v1 = self.model.infer_vector(test_data_list)
        print("V1_infer", v1)

    def get_vocab_size(self):
        return len(self.w2v.vocab)

    def most_similar_words(self, word):
        try:
            similar = self.w2v.most_similar(word)
            words = list((w[0] for w in similar))
            return words
        except KeyError as e:
            return str(e)

    def subtract_from_vectors(self, term1, term_to_remove, term2):
        try:
            similar = self.w2v.most_similar(positive=[term1, term2], negative=[term_to_remove], topn=1)
            words = list((w[0] for w in similar))
            return words
        except KeyError as e:
            return str(e)
    def log(self):
        with open(self.path_logs + "analysis/" + str(datetime.now())[:10] + self.model_name + ".txt", 'w') as fp:
            title = "Model: "+self.model_name
            print("\n"+title+"\n"+("-"*(len(title)+1)))
            print("Saved Analysis log for " + self.model_name)
            fp.write('Vocab Size: ' + str(self.get_vocab_size()) + "\n")
            # fp.write('Vocab Size: '+str(self.get_vocab_size())+"\n")
            if self.type == 'd2v':
                fp.write('Document Count:' + str(Doc2VecAnalysis.get_number_of_docs(self)) + "\n")
            fp.write('Apple: ' + str(self.most_similar_words('apple')) + "\n")
            fp.write('Google: ' + str(self.most_similar_words('google')) + "\n")
            fp.write('Microsoft: ' + str(self.most_similar_words('microsoft')) + "\n")
            fp.write('Tesla: ' + str(self.most_similar_words('tesla')) + "\n")
            fp.write('Equity: ' + str(self.most_similar_words('equity')) + "\n")
            fp.write('Trading: ' + str(self.most_similar_words('trading')) + "\n")
            fp.write('Market: ' + str(self.most_similar_words('market')) + "\n")
            fp.write('{0} + {1} - {2} = '.format('France', 'Paris', 'Italy') + str(
                self.subtract_from_vectors('france', 'paris', 'italy')) + "\n")
            fp.write('{0} + {1} - {2} = '.format('Big', 'Bigger', 'Small') + str(
                self.subtract_from_vectors('big', 'bigger', 'small')) + "\n")
            fp.write('{0} + {1} - {2} = '.format('Miami', 'Florida', 'Berlin') + str(
                self.subtract_from_vectors('miami', 'florida', 'berlin')) + "\n")
            fp.write('{0} + {1} - {2} = '.format('Microsoft', 'Windows', 'Google') + str(
                self.subtract_from_vectors('microsoft', 'windows', 'google')) + "\n")


class Doc2VecAnalysis(Analysis):
    def get_number_of_docs(self):
        return len(self.model.docvecs)

    def most_similar_documents(self, tag):
        similar_doc = self.model.docvecs.most_similar(tag)
        return similar_doc

    def print_vector_by_tag(self, tag):
        return self.model.docvecs[tag]

    def print_vector_by_prefix(self, prefix_string):
        user_docs = [self.model.docvecs[tag] for tag in self.model.docvecs.offset2doctag if
                     tag.startswith(prefix_string)]
        # for doc_id in user_docs:
        #     print(self.model.docvecs[doc_id])
        return user_docs

