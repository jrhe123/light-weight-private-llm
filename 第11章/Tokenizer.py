import re
class SimpleTokenizer:
    def __init__(self, filename):
        self.filename = filename
        self.vocab_w2t = None
        self.vocab_t2w = None

        raw_text = ""
        with open(self.filename, "r", encoding="utf-8") as f:
            raw_text = f.read()
        
        all_words = sorted(set(list(raw_text)))
        all_words.extend(["<|unk|>", "<|endoftext|>"])

        self.vocab_w2t = {token: idx for idx, token in enumerate(all_words)}
        self.vocab_t2w = {idx: token for (token, idx) in self.vocab_w2t.items()}

        # for i, (key, val) in enumerate(self.vocab_t2w.items()):
        #     print(key, val)
        #     if i > 10:
        #         break
            
    def encode(self, text):
        ids = []
        #text="你好"
        #text="北京<|endoftext|>"
        words = re.findall(rf'{re.escape("<|endoftext|>")}|.', text)
        words = [w.strip() for w in words if w.strip()]
        words = [ w if w in self.vocab_w2t else "<|unk|>" for w in words]
        ids = [self.vocab_w2t[w] for w in words]
        return ids
    
    def decode(self, ids):
        text = ""
        # for idx in ids:
        #     print(self.vocab_t2w[idx])
        text = text.join([self.vocab_t2w[idx] for idx in ids])
        return text