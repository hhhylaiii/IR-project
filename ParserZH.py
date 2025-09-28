import sys
import re

try:
    import jieba
except:
    print("Error: Requires jieba from https://github.com/fxsjy/jieba. Have you installed it?")
    sys.exit(1)

def _to_half_width(s: str) -> str:
    out = []
    for ch in s:
        code = ord(ch)
        if code == 0x3000:
            code = 0x20
        elif 0xFF01 <= code <= 0xFF5E: 
            code -= 0xFEE0
        out.append(chr(code))
    return "".join(out)


class ParserZH:

    def __init__(self, hmm=True):
        self.hmm = hmm
        jieba.set_dictionary('dict.txt.big')
        self.stopwords = set()
        with open('chinese.stop', 'r', encoding='utf-8') as f:
            for line in f:
                w=line.strip()
                if w:
                    self.stopwords.add(w)

    def clean(self, string):
        """ remove any nasty grammar tokens from string """
        if string is None:
            return ""
        string = _to_half_width(string)
        string = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", " ", string)
        string = re.sub(r"\s+", " ", string)
        return string.strip()
    
    def tokenise(self, string):
        """ break string up into tokens """
        string = self.clean(string)
        if not string:
            return []
        tokens = jieba.lcut(string, cut_all=False, HMM=self.hmm)
        tokens = [t for t in tokens if t.strip()]
        return tokens
    
    def removeStopWords(self, tokens):
        if tokens is None:
            return []
        return [word for word in tokens if word.strip() and (word not in self.stopwords)]
