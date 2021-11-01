from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor


def compute_scores(gts, res):
    '''

    :param gts: Dictionary with the gold captions
    :param res: Dictionary with the produced captions
    :return: Prints the scores of NLG metrics
    '''
    assert sorted(list(gts.keys())) == sorted(list(res.keys()))
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L")
    ]

    for scorer, method in scorers:
        print("Computing", scorer.method(), "...")
        score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                print("%s : %0.1f" % (m, sc * 100))
        else:
            print("%s : %0.1f" % (method, score * 100))

def prepare_captions(gts, res):
    '''

    :param gts: Dictionary with the gold captions
    :param res: Dictionary with the produced captions
    :return: Dictionaries with the processed gold and produced captions
    '''
    assert list(gts.keys()) == list(res.keys())

    for key in list(gts.keys()):
        g_caption = gts[key]
        r_caption = res[key]

        # Remove special tokens
        if g_caption.split()[0] == "start":
            g_caption = g_caption.replace("start", "", 1)
        if g_caption.split()[-1] == "end":
            g_caption = g_caption.replace("end", "", 1)
        if "startcaption" in g_caption.split():
            g_caption = g_caption.split()
            g_caption = " ".join(g_caption[g_caption.index("startcaption") + 1:])

        gts[key] = " ".join(g_caption.split(" newsentence "))
        res[key] = " ".join(r_caption.split(" newsentence "))

    # Bring dictionaries to pycocoeval format
    gts = {k: [v] for k, v in gts.items()}
    res = {k: [v] for k, v in res.items()}

    return gts, res