import re
import sys

try:
    import jieba
    import jieba.analyse
except:
    print("Error: Requires jieba from https://github.com/fxsjy/jieba. Have you installed it?")
    sys.exit()

