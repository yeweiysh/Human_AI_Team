import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


def plotLossLineGraph(OptionLoss, AgentLoss, Loss, modelName):
    plt.plot(range(0,len(OptionLoss)), OptionLoss, label= "Option Loss")
    plt.plot(range(0,len(AgentLoss)), AgentLoss, label= "Agent Loss")
    plt.plot(range(0,len(Loss)), Loss, label= "Total Loss")
    plt.title(modelName+ " Model Loss values for all teams")
    plt.legend()
    plt.xlabel("Team")
    plt.ylabel("Loss Value")
    # plt.savefig(modelName+".jpg")
    plt.show()
    plt.clf()

def plotAgentUse(agent_number, team_id):
    plt.plot(range(0,len(agent_number)), agent_number)
    plt.title("Team id: " + str(team_id))
    plt.xlabel("index")
    plt.ylabel("Agent ID")
    plt.xticks(np.arange(0, len(agent_number), step=1))
    plt.yticks(np.arange(0, 4, step=1))
    plt.savefig(str(team_id)+".png")
    plt.show()
    plt.clf()

def scatterplotModel(OptionLoss, AgentLoss, modelName):
    # plt.xlim(0.0, 1.2)
    # plt.ylim(0.0, 3)
    plt.scatter(AgentLoss, OptionLoss, s=4, c=("blue"), alpha=0.5)
    # plt.title('Scatter plot-'+modelName)
    plt.xlabel('Agent Loss')
    plt.ylabel('Option Loss')
    # plt.savefig('Scatterplot-'+modelName)
    plt.show()
    plt.clf()

def histogramPlot(data, modelName):
    mu = np.mean(data[0]) # mean of distribution
    sigma = np.std(data[0]) # standard deviation of distribution
    num_bins = 10
    # the histogram of the data
    n, bins, patches = plt.hist(data[0], num_bins, facecolor='blue', alpha=0.5, label = 'Option Loss')
    # plt.xlim(0.0, 3)
    # plt.ylim(0.0, 5)
    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'b-')

    mu = np.mean(data[1]) # mean of distribution
    sigma = np.std(data[1]) # standard deviation of distribution
    num_bins = 10
    # the histogram of the data
    n, bins, patches = plt.hist(data[1], num_bins, facecolor='red', alpha=0.5, label = 'Agent Loss')
    # plt.xlim(0.0, 3)
    # plt.ylim(0.0, 5)
    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r-')


    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    # plt.title('Histogram - '+modelName)
    plt.legend(loc='upper right')
    # # Tweak spacing to prevent clipping of ylabel
    # plt.subplots_adjust(left=0.15)
    # plt.savefig('Histogram-'+modelName+'-'+lossType)
    plt.show()
    plt.clf()


def histPlot(data, modelName):
    mu = np.mean(data) # mean of distribution
    sigma = np.std(data) # standard deviation of distribution
    num_bins = 10
    # the histogram of the data
    n, bins, patches = plt.hist(data, num_bins, facecolor='blue', alpha=0.5, label = 'Simultaneous')
    # plt.xlim(0.0, 3)
    # plt.ylim(0.0, 5)
    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'b-')


    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()
    plt.clf()

def groupVSbestHumanAccuracy(ratio):
    index = []
    for i in range(len(ratio)):
        index.append(i+1)

    plt.bar(index, ratio, align='center',color='blue')
    plt.plot((1, 30), (1, 1), 'red')
    # plt.xlim(0, len(ratio))
    plt.ylim(0.0, 1.5)
    plt.xlabel('Team number')
    plt.ylabel('Group Accuracy / Best Machine Accuracy')
    # plt.title('Group Accuracy vs Best Human Accuracy Ratio')
    plt.savefig('groupVSbestMachineAccuracy')
    plt.show()
    plt.clf()

def groupAccuracyOverTime(group_accuracy_over_time):
    index = []
    acc = []
    # for i in range(4,len(group_accuracy_over_time),5):
    for i in range(len(group_accuracy_over_time)):
        index.append(i+1)
        acc.append(group_accuracy_over_time[i])

    # plt.plot(range(len(group_accuracy_over_time)), group_accuracy_over_time)
    plt.plot(index, acc, '-bo')
    plt.xlabel('Question number')
    plt.ylabel('Group accuracy')
    # plt.title('Group Accuracy over Time')
    plt.savefig('groupAccuracyOverTime')
    plt.show()
    plt.clf()
