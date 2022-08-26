import pandas as pd
import seaborn as sns
def createGpuGraph():
    dfGpu = pd.read_excel('tyTimeListGPU.xlsx')
    y = dfGpu.values
    graphGpu = sns.histplot(y)
    figureGpu = graphGpu.get_figure()
    figureGpu.savefig("tyTimeGraphGPUTemp.png")


def createNNGraph():
    dfNN = pd.read_excel('tyTimeListNN.xlsx')
    y = dfNN.values
    graphNN = sns.histplot(y)
    figureNN = graphNN.get_figure()
    figureNN.savefig("tyTimeGraphNNTemp.png")

if __name__ == "__main__":
    createNNGraph()
    #createGpuGraph()

