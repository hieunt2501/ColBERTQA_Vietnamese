from vncorenlp import VnCoreNLP
from colbert.text_processor.rulebased import RuleBasedPostprocessor

class TextProcessor:
    def __init__(self):
        self.annotator = VnCoreNLP("./colbert/VnCoreNLP/VnCoreNLP-1.1.1.jar", 
                                    annotators="wseg", 
                                    max_heap_size='-Xmx2g')
        self.rb_processor = RuleBasedPostprocessor()

    def process(self, text):
        sentences = self.annotator.tokenize(text)
        annotated_text = [words for sentence in sentences for words in sentence]

        return self.rb_processor.correct(" ".join(annotated_text))