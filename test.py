from sklearn.ensemble import RandomForestClassifier
import numpy as np

domainlist = []
testlist = []

class Domain:
    def __init__(self, _name, _label):
        self.name = _name
        self.label = _label
        self.namelength = len(_name)

        dig = alp = 0
        for i in _name:
            if i.isalpha():
                alp += 1
            else:
                dig += 1

        self.NumInDomain = dig
        self.Entropy = dig/alp

    def returnData(self):
        return [self.namelength,self.NumInDomain,self.Entropy]

    def returnLabel(self):
        if self.label == "dga":
            return 0
        else:
            return 1

    def changeLabel(self):
        self.label = "notdga"

    def returnResult(self):
        return self.name + ","+self.label +"\n"

def InitData(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            tokens = line.split(",")
            name = tokens[0]
            label = tokens[1]

            domainlist.append(Domain(name, label,))


def InitTest(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            tokens = line.split(",")
            name = tokens[0]
            label = "dga"

            testlist.append(Domain(name, label,))

def main():
    InitData("train.txt")
    InitTest("test.txt")

    featureMatrix = []
    labelList = []

    for item in domainlist:
        featureMatrix.append(item.returnData())
        labelList.append(item.returnLabel())

    clf = RandomForestClassifier(random_state=0)
    clf.fit(featureMatrix, labelList)
    with open("result.txt", 'w') as f:
        for item in testlist:
            if(clf.predict([item.returnData()])[0] == 1):
                item.changeLabel()
            f.writelines(item.returnResult())



if __name__ == '__main__':
    main()

