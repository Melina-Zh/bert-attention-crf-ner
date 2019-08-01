from xml.dom.minidom import parse
import xml.dom.minidom
import os



def xml2xt(path1, name):
    nota=['B-ap', 'I-ap', 'O']
    DOMTree = xml.dom.minidom.parse(path1)
    collection = DOMTree.documentElement
    sentences = collection.getElementsByTagName("sentence")
    output = open(name + '_set.txt', 'w')
    texts = []
    labels_a = []
    labels_p = []
    for x, sentence in enumerate(sentences):
        text = sentence.getElementsByTagName('text')[0]
        text = text.childNodes[0].data
        text = text.lower()
        words = text.split(' ')
        label_a = ['2'] * len(words)

        aspects = sentence.getElementsByTagName('aspectTerms')
        if len(aspects) > 0:
            aspects = aspects[0].getElementsByTagName('aspectTerm')
            for aspect in aspects:
                s = aspect.getAttribute("from")
                s = int(s)
                e = aspect.getAttribute("to")
                e = int(e)
                l = 0
                for i, word in enumerate(words):
                    if l <= s:
                        if l + len(word) > s:
                            label_a[i] = '0'
                    elif (l > s) and (l < e):
                        label_a[i] = '1'
                    l += len(word)
                    l += 1

            res = text + "|||" + (" ".join(nota[int(j)] for j in label_a)) + '\n'
            output.write(res)




xml2xt('../data/ABSA-SemEval2014/Restaurants_Train_v2.xml', '../data/ABSA-SemEval2014/Restaurants_Train_v2')
#xml_to_txt('Restaurants_Test_Data_phaseB.xml', 'test_restaurant', 'test')