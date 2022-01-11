import pandas as pd
import numpy as np
import fileinput
import json
from scipy.stats import beta
from scipy import stats
import matplotlib.pyplot as plt
import re
import networkx as nx
import math
import random
import pickle
import AnalysisFunctions as plf
from scipy.stats import wilcoxon
from statistics import mean
from scipy.stats import pearsonr
from cpt_valuation import evaluateProspectVals
from sklearn.metrics import precision_recall_fscore_support
import sys
from numpy import linalg as LA
from networkx.algorithms.dag import is_aperiodic
from networkx.algorithms.components import is_strongly_connected
# from networkx.algorithms.centrality import eigenvector_centrality
from itertools import groupby

class HumanDecisionModels:
    def __init__(self,teamId,directory):

        #Constants
        self.numQuestions = 45
        self.trainingSetSize = 30
        self.testSetSize = 15
        self.numAgents = 4

        self.numCentralityReports = 9

        self.c = 4
        self.e = -1
        self.z = -1

#         Other Parameters
        self.influenceMatrixIndex = 0
        self.machineUseCount = [-1, -1, -1, -1]
        self.firstMachineUsage = [-1, -1, -1, -1]

        # Preloading of the data
        eventLog = pd.read_csv(directory+"event_log.csv", sep=',',quotechar="|", names=["id","event_type","event_content","timestamp","completed_task_id","sender_subject_id","receiver_subject_id","session_id","sender","receiver","extra_data"])

        teamSubjects = pd.read_csv(directory+"team_has_subject.csv",sep=',',quotechar="|",names=["id","teamId","sender_subject_id"]).drop('id',1)

        elNoMessage =  eventLog[(eventLog['event_type'] == "TASK_ATTRIBUTE")]

        elNoMessage["sender_subject_id"] = pd.to_numeric(elNoMessage["sender_subject_id"])

        eventLogWithTeam = pd.merge(elNoMessage, teamSubjects, on='sender_subject_id', how='left')
        eventLogTaskAttribute = eventLogWithTeam[(eventLogWithTeam['event_type'] == "TASK_ATTRIBUTE") & (eventLogWithTeam['teamId'] == teamId)]
        #Extract data from event_content column
        newEventContent = pd.DataFrame(index=np.arange(0, len(eventLogTaskAttribute)), columns=("id","stringValue", "questionNumber","questionScore","attributeName"))
        self.questionNumbers = list()

        for i in range(len(eventLogTaskAttribute)):
            newEventContent.id[i] = eventLogTaskAttribute.iloc[i]["id"]
            newEventContent.stringValue[i] = eventLogTaskAttribute.iloc[i]["event_content"].split("||")[0].split(":")[1].replace('"', '')
            newEventContent.questionNumber[i] = eventLogTaskAttribute.iloc[i]["event_content"].split("||")[1].split(":")[1]
            if newEventContent.questionNumber[i] not in self.questionNumbers:
                self.questionNumbers.append(newEventContent.questionNumber[i])
            newEventContent.questionScore[i] = eventLogTaskAttribute.iloc[i]["event_content"].split("||")[3].split(":")[1]
            newEventContent.attributeName[i] =eventLogTaskAttribute.iloc[i]["event_content"].split("||")[2].split(":")[1]

        self.questionNumbers = self.questionNumbers[1 :]
        eventLogWithAllData = pd.merge(eventLogTaskAttribute,newEventContent,on='id', how ='left')

        self.machineAsked = eventLogWithAllData[eventLogWithAllData['extra_data'] == "AskedMachine"]
        self.machineAskedQuestions = list()
        for i in range(len(self.machineAsked)):
            self.machineAskedQuestions.append(int(float(self.machineAsked.iloc[i]['questionNumber'])))

        # Load correct answers
        with open(directory+"jeopardy45.json") as json_data:
                d = json.load(json_data)
        self.correctAnswers = list()
        self.options = list()

        for i in range(0, self.numQuestions):
            self.correctAnswers.append(d[int(float(self.questionNumbers[i]))-1]['Answer'])
            self.options.append(d[int(float(self.questionNumbers[i]))-1]['value'])


        allIndividualResponses = eventLogWithAllData[eventLogWithAllData['extra_data'] == "IndividualResponse"]

        self.lastIndividualResponsesbyQNo = allIndividualResponses.groupby(['sender', 'questionNumber'], as_index=False, sort=False).last()

        # Compute the group answer of the team per question
        submissions = eventLogWithAllData[(eventLogWithAllData['extra_data'] == "IndividualResponse") | (eventLogWithAllData['extra_data'] == "GroupRadioResponse") ]
        individualAnswersPerQuestion = submissions.groupby(["questionNumber","sender_subject_id"], as_index=False, sort=False).tail(1)

        self.groupSubmission = pd.DataFrame(index=np.arange(0, len(self.questionNumbers)), columns=("questionNumber","groupAnswer"))
        for i in range(0, self.numQuestions):
            ans = ""
            consensusReached = True
            for j in range(0,len(individualAnswersPerQuestion)):
                if (individualAnswersPerQuestion.iloc[j].loc["questionNumber"] == self.questionNumbers[i]):
                    if not ans:
                        ans = individualAnswersPerQuestion.iloc[j].loc["stringValue"]

                    elif (ans != individualAnswersPerQuestion.iloc[j].loc["stringValue"]):
                        consensusReached = False
                        break

            self.groupSubmission.questionNumber[i] = self.questionNumbers[i]
            if (consensusReached):
                self.groupSubmission.groupAnswer[i] = ans
            else:
                self.groupSubmission.groupAnswer[i] = "Consensus Not Reached"

        # Define teammember order
        subjects = pd.read_csv(directory+"subject.csv", sep=',',quotechar="|", names=["sender_subject_id","externalId","displayName","sessionId","previousSessionSubject"])
        teamWithSujectDetails = pd.merge(teamSubjects, subjects, on='sender_subject_id', how='left')
        self.teamMember = teamWithSujectDetails[(teamWithSujectDetails['teamId'] == teamId)]['displayName']
        self.teamSize = len(self.teamMember)
        self.teamArray = list()

        for i in range(self.teamSize):
            self.teamArray.append(self.teamMember.iloc[i])
        
        #         Pre-experiment Survey
        preExperimentData = eventLogWithAllData[eventLogWithAllData['extra_data'] == "RadioField"]
        self.preExperimentRating = list()
        for i in range(0,self.teamSize):
            self.preExperimentRating.append(0)
            if len(preExperimentData[(preExperimentData['sender'] == self.teamMember.iloc[i]) & (preExperimentData['attributeName'] == "\"surveyAnswer0\"")])>0:
                self.preExperimentRating[-1]+=(float(preExperimentData[(preExperimentData['sender'] == self.teamMember.iloc[i]) & (preExperimentData['attributeName'] == "\"surveyAnswer0\"")]['stringValue'].iloc[0][0:1]))
            if len(preExperimentData[(preExperimentData['sender'] == self.teamMember.iloc[i]) & (preExperimentData['attributeName'] == "\"surveyAnswer1\"")]) >0:
                self.preExperimentRating[-1]+=(float(preExperimentData[(preExperimentData['sender'] == self.teamMember.iloc[i]) & (preExperimentData['attributeName'] == "\"surveyAnswer1\"")]['stringValue'].iloc[0][0:1]))
            if len(preExperimentData[(preExperimentData['sender'] == self.teamMember.iloc[i]) & (preExperimentData['attributeName'] == "\"surveyAnswer2\"")])>0:
                self.preExperimentRating[-1]+=(float(preExperimentData[(preExperimentData['sender'] == self.teamMember.iloc[i]) & (preExperimentData['attributeName'] == "\"surveyAnswer2\"")]['stringValue'].iloc[0][0:1]))
            self.preExperimentRating[-1]/=15


        # Extracting Machine Usage Information
        self.machineUsed = np.array([False, False, False, False] * self.numQuestions).reshape((self.numQuestions, 4))
        for i in range(self.numQuestions):
            if int(float(self.questionNumbers[i])) in self.machineAskedQuestions:
                indxM = self.machineAskedQuestions.index(int(float(self.questionNumbers[i])))
                k = self.teamArray.index(self.machineAsked['sender'].iloc[indxM])
                self.machineUsed[i][int(k)] = True
        


        # print("machineUsed: ")
        # print(self.machineUsed)



        self.teamScore = list()
        self.computeTeamScore()

#         Extract Influence Matrices
        self.agentRatings = list()
        self.memberInfluences = list()
        mInfluences = [0 for i in range(self.teamSize)]
        aRatings = [0 for i in range(self.teamSize)]
        count = 0
        influenceMatrices = eventLogWithAllData[(eventLogWithAllData['extra_data'] == "InfluenceMatrix")]
        influenceMatrixWithoutUndefined = influenceMatrices[~influenceMatrices['stringValue'].str.contains("undefined")]
        finalInfluences = influenceMatrixWithoutUndefined.groupby(['questionScore', 'sender'], as_index=False, sort=False).last()

        for i in range(len(finalInfluences)):
            count +=1
            aR = list()
            mI = list()
            idx = self.teamArray.index(finalInfluences.iloc[i]['sender'])
            for j in range(0, self.teamSize):
                temp = finalInfluences.iloc[i]['stringValue']
#                 Fill missing values
                xy = re.findall(r'Ratings(.*?) Member', temp)[0].split("+")[j].split("=")[1]
                if(xy==''):
                    xy = '0.5'
                yz= temp.replace('"', '')[temp.index("Influences ")+10:].split("+")[j].split("=")[1]
                if(yz == ''):
                    yz = '25'
                aR.append(float(xy))
                mI.append(int(round(float(yz))))
            aRatings[idx]=aR
            mInfluences[idx]=mI
            if(count%self.teamSize == 0):
                self.memberInfluences.append(mInfluences)
                mInfluences = [0 for i in range(self.teamSize)]
                self.agentRatings.append(aRatings)
                aRatings = [0 for i in range(self.teamSize)]

        # Hyperparameters for expected performance (Humans and Agents) - TODO
        self.alphas = [1,1,1,1,1,1,1,1]
        self.betas = np.ones(8, dtype = int)

        #vector c
        self.centralities = [[] for _ in range(self.numQuestions)]

        self.actionTaken = list()
        self.computeActionTaken()

    def computeTeamScore(self):
        self.teamScore.append(0)
        for i in range(0,self.numQuestions):
            if self.groupSubmission.groupAnswer[i]!=self.correctAnswers[i]:
                self.teamScore[i]+=self.z
            else:
                self.teamScore[i]+=self.c
            if len(np.where(self.machineUsed[i] == True)[0])!=0:
                self.teamScore[i]+=self.e
            self.teamScore.append(self.teamScore[i])
        self.teamScore = self.teamScore[:-1]

    def updateAlphaBeta(self, i, valueSubmitted, correctAnswer):
        if (valueSubmitted == correctAnswer):
            self.alphas[i]+=1
        else:
            self.betas[i]+=1

    def naiveProbability(self, questionNumber, idx):
        expectedPerformance = list()
        individualResponse = list()
        probabilities = list()
        human_accuracy = list()

        machine_accuracy = [None for _ in range(self.numAgents)]
        group_accuracy = 0

        #Save human expected performance based
        for i in range(0,self.teamSize):
            expectedPerformance.append(beta.mean(self.alphas[i],self.betas[i]))
            individualResponse.append(self.lastIndividualResponsesbyQNo[(self.lastIndividualResponsesbyQNo["questionNumber"] == questionNumber) & (self.lastIndividualResponsesbyQNo["sender"] == self.teamMember.iloc[i])]["stringValue"].any())
            self.updateAlphaBeta(i,self.lastIndividualResponsesbyQNo[(self.lastIndividualResponsesbyQNo["questionNumber"] == questionNumber) & (self.lastIndividualResponsesbyQNo["sender"] == self.teamMember.iloc[i])]["stringValue"].any(),self.correctAnswers[idx])

            ans = self.lastIndividualResponsesbyQNo[(self.lastIndividualResponsesbyQNo["questionNumber"] == questionNumber) & (self.lastIndividualResponsesbyQNo["sender"] == self.teamMember.iloc[i])]["stringValue"].any()
            if ans == self.correctAnswers[idx]:
                human_accuracy.append(1)
            else:
                human_accuracy.append(0)

        if (self.groupSubmission["groupAnswer"].iloc[idx] == self.correctAnswers[idx]):
            group_accuracy = 1

        indxQ = -1
        anyMachineAsked = False
        if(int(float(questionNumber)) in self.machineAskedQuestions):
            indxQ = self.machineAskedQuestions.index(int(float(questionNumber)))
            sender = self.machineAsked['sender'].iloc[indxQ]
            k = self.teamArray.index(sender)
            # print("k")
            # print(k)
            anyMachineAsked = True

        # Add expected Performance for Agents
        for i in range(self.teamSize, self.teamSize+self.numAgents):
            expectedPerformance.append(0.5 + 0.5 * beta.mean(self.alphas[i],self.betas[i]))
            # update alpha beta for that machine

        #Update machine accuracy
        if(anyMachineAsked):
            machineAnswer = self.machineAsked['event_content'].iloc[indxQ].split("||")[0].split(":")[2].replace('"', '').split("_")[0]
            self.updateAlphaBeta(self.getAgentForHuman(k), machineAnswer, self.correctAnswers[idx])
            self.machineUseCount[k]+=1

            if self.firstMachineUsage[k] == -1:
                self.firstMachineUsage[k] = idx

            if (machineAnswer == self.correctAnswers[idx]):
                machine_accuracy[k] = 1
            else:
                machine_accuracy[k] = 0

            # machine_accuracy[k] = 1

   
        # Conditional Probability
        # Do a bayes update
        denominator = 0
        numerator = [1. for _ in range(len(self.options[idx]))]
        prob_class = 0.25
        prob_rsep = 0
        prob_class_responses = [None for _ in range(len(self.options[idx]))]
        prob_resp_given_class = [None for _ in range(len(self.options[idx]))]

        for opt_num in range(0,len(self.options[idx])):
            prob_resp = 0
            numerator = prob_class
            for person_num in range(0,self.teamSize):
                if individualResponse[person_num] == self.options[idx][opt_num]:
                    numerator *= expectedPerformance[person_num]
                else:
                    numerator *= (1 - expectedPerformance[person_num])/3
                prob_resp += numerator
            prob_resp_given_class[opt_num] = numerator
        prob_class_responses = [(prob_resp_given_class[i]/sum(prob_resp_given_class)) for i in range(0,len(prob_resp_given_class))]

        # print("expectedPerformance: ")
        # print(expectedPerformance)
        #ANSIs this updating agent probabilities?
        # for i in range(self.teamSize):
        #     probabilities.append(expectedPerformance[self.teamSize+i])
        l1 = 0
        for i in range(self.teamSize):
            l1 += expectedPerformance[self.teamSize+i]

        for i in range(self.teamSize):
            probabilities.append(1.0 * expectedPerformance[self.teamSize+i]/l1)

        #8 probability values returned
        # first set is for options (sums to 1)

        assert(sum(prob_class_responses) > 0.999 and sum(prob_class_responses) < 1.001)
        #second set is for machines
        prob_all_class_responses = prob_class_responses + [1.0 * expectedPerformance[self.getAgentForHuman(k)] / l1 for k in range(self.teamSize)]
        
        # print("expectedPerformance: ")
        # print(expectedPerformance)
        # print("prob_all_class_responses: ")
        # print(prob_all_class_responses)

        return expectedPerformance, prob_all_class_responses,human_accuracy,group_accuracy,machine_accuracy


    def updateCentrality(self, influenceMatrixIndex):
        #Compute Eigen Vector Centrality for Humans
        influM = np.zeros((self.teamSize,self.teamSize))
        for i in range(0,self.teamSize):
            ss = 0
            for j in range(0,self.teamSize):
                influM[i][j] = self.memberInfluences[influenceMatrixIndex][i][j]/100
                #influM[i][j] += sys.float_info.epsilon
                ss += influM[i][j]

            for j in range(0,self.teamSize):
                influM[i][j] /= ss


        agentR = np.zeros((self.teamSize,self.teamSize))
        for i in range(0,self.teamSize):
            ss = 0
            for j in range(0,self.teamSize):
                agentR[i][j] = self.agentRatings[influenceMatrixIndex][i][j]
                ss += agentR[i][j]
            if ss != 0:
                for j in range(0,self.teamSize):
                    agentR[i][j] /= ss


        graph = nx.DiGraph()
        for i in range(0,self.teamSize):
            for j in range(0,self.teamSize):
                graph.add_edge(i,j,weight=influM[i][j])


        graph1 = nx.DiGraph()
        for i in range(0,self.teamSize):
            for j in range(0,self.teamSize):
                graph1.add_edge(i,j,weight=agentR[i][j])

        aperiodic = is_aperiodic(graph1)
        connected = is_strongly_connected(graph1)

        W1 = nx.adjacency_matrix(graph1)
        agent_centralities = nx.eigenvector_centrality(graph1, max_iter=1000, weight="weight")
        eig_centrality1 = list(agent_centralities.values())
        summ = 0
        for eigv in eig_centrality1:
            summ += eigv
        for i in range(4):
            eig_centrality1[i] /= summ


        W = nx.adjacency_matrix(graph)
        # print(W.todense())
        # eigenValues, eigenVectors = np.linalg.eigh(W.todense())
        # print(eigenValues)

        human_centralities = nx.eigenvector_centrality(graph, weight="weight")

        # aa=eigenvector_centrality(graph, weight="weight")
        eig_centrality = list(human_centralities.values())
        summ = 0
        for eigv in eig_centrality:
            summ += eigv
        for i in range(4):
            eig_centrality[i] /= summ


        # print(np.argsort(eig_centrality))
        # print(list(aa.values()))
        # print(eig_centrality)
        #largest = max(nx.strongly_connected_components(graph), key=len)

        #Compute expected performance for machines

        """
        for i in range(0,self.teamSize):
            numerator = 0
            denominator = 0
            for j in range(0,self.teamSize):
                numerator+= self.centralities[j] * self.agentRatings[influenceMatrixIndex][j][i]
                denominator+= self.centralities[j]
            self.centralities.update({self.teamSize+i:numerator/denominator})

        """
        #Check that we have the correct total influence
        for i in range(self.teamSize):
            assert(sum(self.memberInfluences[influenceMatrixIndex][i][j] for j in range(self.numAgents)) == 100)

        #Make a probability

        agent_weighted_centrality_perf = [None for _ in range(self.numAgents)]
        '''
        for i in range(self.numAgents):
            agent_weighted_centrality_perf[i] = sum([self.memberInfluences[influenceMatrixIndex][i][j]/100. for j in range(self.numAgents)])
        '''
        centralities_as_list = [value for value in human_centralities.values()]
        for i in range(self.numAgents):
            agent_weighted_centrality_perf[i] = sum([centralities_as_list[j]*self.agentRatings[influenceMatrixIndex][j][i] for j in range(self.numAgents)])/sum(centralities_as_list)

        for question_num in range(self.influenceMatrixIndex*5 ,(self.influenceMatrixIndex+1)*5):
            self.centralities[question_num] = centralities_as_list + agent_weighted_centrality_perf

        #Move to next influence matrix
        self.influenceMatrixIndex+=1

        return np.asarray(W1.todense()), np.asarray(eig_centrality1), aperiodic, connected

    def calculatePerformanceProbability(self, questionNumber, idx):
        probabilities = list()
        probabilities = [0 for _ in range(self.teamSize + self.numAgents)]

        person_response = []

        for i in range(0,self.teamSize):
            #print("i:")
            #print(i)
            #print(self.centralities[idx][i])
            individulResponse = self.lastIndividualResponsesbyQNo[(self.lastIndividualResponsesbyQNo["questionNumber"] == questionNumber) & (self.lastIndividualResponsesbyQNo["sender"] == self.teamMember.iloc[i])]["stringValue"].any()
            # print(individulResponse)
            person_response.append(individulResponse)
            index = self.options[idx].index(individulResponse)
            #print("index:")
            #print(index)
            #print(self.centralities[idx][i])
            probabilities[index] += self.centralities[idx][i]
        #print("probability:")
        #print(probabilities)
        # Normalize the probabilties
        totalProbability = sum(probabilities)
        probabilities[:] = [x / totalProbability for x in probabilities]


        # Add expected Performance for Agents
        for i in range(0, self.numAgents):
            #which agents should have a centrality of 1?
            if self.centralities[idx][self.getAgentForHuman(i)] == 1:
                probabilities[self.getAgentForHuman(i)] = self.centralities[idx][self.getAgentForHuman(i)]
            #which agents should have a positive centrality
            elif self.centralities[idx][i+self.teamSize] >= 0:
                probabilities[self.getAgentForHuman(i)] = self.centralities[idx][self.getAgentForHuman(i)]
            else:
                assert(False) # no negative centralities allowed
        
        l1 = 0
        for i in range(4,8):
            l1 += probabilities[i]

        for i in range(4,8):
            probabilities[i] /= l1 

        return probabilities, person_response

    def calculateModelAccuracy(self,perQuestionRewards,probabilities,idx):
        highestRewardOption = max(perQuestionRewards[0:4])
        highestRewardAgent = max(perQuestionRewards[4:8])
        modelAccuracy = 0
        count = 0

        if highestRewardOption >= highestRewardAgent:
            for i in range(0,self.teamSize):
                if highestRewardOption == perQuestionRewards[i] and self.options[idx][i]==self.correctAnswers[idx]:
                    count+=1
                    modelAccuracy = 1
            modelAccuracy = modelAccuracy * count / (perQuestionRewards[0:4].count(highestRewardOption))
        else:
            for i in range(self.teamSize,self.teamSize*2):
                if highestRewardAgent == perQuestionRewards[i]:
                    modelAccuracy += probabilities[i] * (perQuestionRewards[4:8].count(highestRewardAgent))
        return modelAccuracy

    # Expected rewards for (all options + all agents)
    def calculateExpectedReward(self, probabilities):
        perQuestionRewards = list()
        for j in range(0,self.teamSize):
            perQuestionRewards.append(self.c*probabilities[j] + (self.z)*(1-probabilities[j]))

        for j in range(0,self.teamSize):
            perQuestionRewards.append((self.c+self.e)*probabilities[self.getAgentForHuman(j)] + (self.z+self.e)*(1-probabilities[self.getAgentForHuman(j)]))

        # print("probabilities: ")
        # print(probabilities)
        # print("perQuestionRewards: ")
        # print(perQuestionRewards)
        return perQuestionRewards

    def calculateRewards(self):
        rewardsNB1 = list()
        probabilitiesNB1 = list()
        probabilitiesRANDOM = list()
        group_accuracies = list()
        expectedP = list()
        group_accuracy_per_question = list() # for each question
        all_accuracy = []
        consult_agent = []
        consult_agent_id = []
        agent_response = []
        agent_acc_for_dt2 = []
        human_acc_for_dt2 = []
        group_acc_for_dt2 = []
        non_consensus_for_dt2 = []

        h1 = list()
        h2 = list()
        h3 = list()
        h4 = list()
        a1 = list()
        a2 = list()
        a3 = list()
        a4 = list()
        random_p = [0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125]
        human_p = []
        # Compute Reward for NB1
        for i in range(0,self.numQuestions):
            expectedPerformance, all_probabilities, human_accuracy, group_accuracy, machine_accuracy = self.naiveProbability(self.questionNumbers[i],i)
            group_accuracy_per_question.append(group_accuracy)
            h1.append(human_accuracy[0])
            h2.append(human_accuracy[1])
            h3.append(human_accuracy[2])
            h4.append(human_accuracy[3])
            a1.append(machine_accuracy[0])
            a2.append(machine_accuracy[1])
            a3.append(machine_accuracy[2])
            a4.append(machine_accuracy[3])

            if (machine_accuracy[0] != None) or (machine_accuracy[1] != None) or (machine_accuracy[2] != None) or (machine_accuracy[3] != None):
                consult_agent.append(1)
            else:
                consult_agent.append(0)

            if (machine_accuracy[0] == 1) or (machine_accuracy[1] == 1) or (machine_accuracy[2] == 1) or (machine_accuracy[3] == 1):
                agent_response.append(1)
            else:
                agent_response.append(0)

            if (machine_accuracy[0] != None) or (machine_accuracy[1] != None) or (machine_accuracy[2] != None) or (machine_accuracy[3] != None):
                human_acc_for_dt2.append(human_accuracy)
                group_acc_for_dt2.append(group_accuracy)
                if self.actionTaken[i]==-1:
                    non_consensus_for_dt2.append(-1)
                else:
                    non_consensus_for_dt2.append(1)

            if (machine_accuracy[0] != None):
                consult_agent_id.append(0)
                agent_acc_for_dt2.append(machine_accuracy[0])
            elif (machine_accuracy[1] != None):
                consult_agent_id.append(1)
                agent_acc_for_dt2.append(machine_accuracy[1])
            elif (machine_accuracy[2] != None):
                consult_agent_id.append(2)
                agent_acc_for_dt2.append(machine_accuracy[2])
            elif (machine_accuracy[3] != None):
                consult_agent_id.append(3)
                agent_acc_for_dt2.append(machine_accuracy[3])

            probabilitiesNB1.append(all_probabilities)
            probabilitiesRANDOM.append(random_p)

            rewardsNB1.append(self.calculateExpectedReward(all_probabilities))
            expectedP.append(expectedPerformance)

        
        human_p.append(h1)
        human_p.append(h2)
        human_p.append(h3)
        human_p.append(h4)

        all_accuracy.append(consult_agent)
        all_accuracy.append(group_accuracy_per_question)
        all_accuracy.append(agent_response)

        #Compute Reward for CENT1 model
        rewardsCENT1 = list()
        probabilitiesCENT1 = list()
        #largest_components = []
        if_aperiodic = []
        if_connected = []
        adjacency = []
        eig_cen = []
        for i in range(0,self.numCentralityReports):
            adj, eigenvector_centrality, aperiodic, connected = self.updateCentrality(self.influenceMatrixIndex)
            #largest_components.append(len(largest))
            if_aperiodic.append(aperiodic)
            if_connected.append(connected)
            adjacency.append(adj)
            eig_cen.append(eigenvector_centrality)

        before_consensus = []
        for i in range(0,self.numQuestions):
            probabilities, person_response = self.calculatePerformanceProbability(self.questionNumbers[i],i)
            if len(set(person_response)) == 1:
                before_consensus.append(1)
            else:
                before_consensus.append(0)
            probabilitiesCENT1.append(probabilities)
            rewardsCENT1.append(self.calculateExpectedReward(probabilities))

        # print("largest component")
        # print(largest_components)
        # print("if aperiodic:")
        # print(if_aperiodic)
        # print("if strongly connected:")
        # print(if_connected)

        sum_before_consensus = []
        # print("before consensus:")
        # print(before_consensus)
        # for i in range(9):
        #     ss = 0
        #     for j in range(i*5,i*5+5):
        #         ss += before_consensus[j]
        #     sum_before_consensus.append(ss)

        # print(sum_before_consensus)

        frobenius = []
        l2 = []
        s_diff = 0

        for i in range(1,len(adjacency)):
            frobenius.append(LA.norm(adjacency[i]-adjacency[i-1], 'fro'))
            l2.append(LA.norm(eig_cen[i]-eig_cen[i-1],ord=2))
            s_diff += LA.norm(eig_cen[i]-eig_cen[i-1],ord=2)

        s_infinity = 0
        avg_dis = []
        for i in range(len(adjacency)):
            adj=adjacency[i]
            ss = 0
            for j in range(len(adj)):
                ss += LA.norm(adj[j,:]-eig_cen[i],ord=2)
            ss /= 4.0
            avg_dis.append(ss)
            s_infinity += ss

        # print("average distance:")
        # for i in range(len(avg_dis)):
        #     avg_dis[i] = round(avg_dis[i],3)
        # print(avg_dis)
        # print("L2 distance:")
        # for i in range(len(l2)):
        #     l2[i] = round(l2[i],3)
        # print(l2)

        # print("distance:")
        # print(s_infinity)
        # print(frobenius)
        # print(l2)
        p_h = [sum(h1)/len(h1), sum(h2)/len(h2), sum(h3)/len(h3), sum(h4)/len(h4)]
        
        best_human_accuracy = max(p_h)

        # best_human_index = np.argmax(np.asarray(p_h))
        # best_human_cen = []
        # mean_cen = []
        # min_cen = []
        # max_cen = []

        # cnt1 = 0
        # cnt2 = 0
        # for eigs in eig_cen:
        #     be = round(eigs[best_human_index],3)
        #     best_human_cen.append(be)
        #     me = round(np.mean(np.asarray(eigs)),3)
        #     lo = round(np.amin(np.asarray(eigs)),3)
        #     hi = round(np.amax(np.asarray(eigs)),3)
        #     mean_cen.append(me)
        #     min_cen.append(lo)
        #     max_cen.append(hi)
        #     if be == hi:
        #         cnt1 += 1
        #     if be >= me:
        #         cnt2 += 1

        # print("best human centrality:")
        # print(best_human_cen)
        # print(mean_cen)
        # print(min_cen)
        # print(max_cen)
        # print(cnt1)
        # print(cnt2)




        # best_human_accuracy = max([sum(h1[30:])/15, sum(h2[30:])/15, sum(h3[30:])/15, sum(h4[30:])/15])

        m1=[]
        m2=[]
        m3=[]
        m4=[]
        for ele in a1:
            if ele != None:
                m1.append(ele)

        for ele in a2:
            if ele != None:
                m2.append(ele)

        for ele in a3:
            if ele != None:
                m3.append(ele)

        for ele in a4:
            if ele != None:
                m4.append(ele)

        best_machine_accuracy = 0.000001
        if len(m1) != 0:
            if sum(m1)/len(m1) >=best_machine_accuracy:
                best_machine_accuracy = sum(m1)/len(m1)

        if len(m2) != 0:
            if sum(m2)/len(m2) >=best_machine_accuracy:
                best_machine_accuracy = sum(m2)/len(m2)

        if len(m3) != 0:
            if sum(m3)/len(m3) >=best_machine_accuracy:
                best_machine_accuracy = sum(m3)/len(m3)

        if len(m4) != 0:
            if sum(m4)/len(m4) >=best_machine_accuracy:
                best_machine_accuracy = sum(m4)/len(m4)

        return expectedP, rewardsNB1,rewardsCENT1, probabilitiesNB1,probabilitiesCENT1, probabilitiesRANDOM, group_accuracy_per_question, best_human_accuracy, best_machine_accuracy, all_accuracy,human_p,before_consensus,agent_acc_for_dt2,human_acc_for_dt2,group_acc_for_dt2,non_consensus_for_dt2,eig_cen

    #--Deprecated--
    def computePTaccuracy(self, pi):
        PTrewards = list()
        for i in range(0,len(pi)):
            PTrewards.append(model.calculateExpectedReward(pi[i]))
        accuracy = list()
        for i in range(0,len(pi)):
            if i==0:
                accuracy.append(self. calculateModelAccuracy(PTrewards[i],pi[i],(i+self.trainingSetSize))/(i+1))
            else:
                accuracy.append((self.calculateModelAccuracy(PTrewards[i],pi[i],(i+self.trainingSetSize)) + (i*accuracy[i-1]))/(i+1))
        return PTrewards, accuracy

    def softmax(self, vec):
        return np.exp(vec) / np.sum(np.exp(vec), axis=0)

        # Called in loss function --Deprecated--
    def newValues(self,values):
        least = min(values)
        values[:] = [i-least for i in values]
        values[:] = [i/sum(values) for i in values]
        return values

    def computeActionTaken(self):
        for i in range(0,self.numQuestions):
            if self.groupSubmission.groupAnswer[i] == "Consensus Not Reached":
                self.actionTaken.append(-1)
            elif int(float(self.questionNumbers[i])) in self.machineAskedQuestions:
                self.actionTaken.append(self.teamSize + np.where(self.machineUsed[i] == True)[0][0])
            else:
                self.actionTaken.append(self.options[i].index(self.groupSubmission.groupAnswer[i]))

#     Computes V1 to V8 for a given question --Deprecated--
    def computeCPT(self,alpha,gamma,probabilities):
        values = list()
        for i in range(0,2*self.teamSize):
            if i<4:
                values.append((math.pow(self.c, alpha) * math.exp(-math.pow(math.log(1/probabilities[i]), gamma)))-(math.pow(math.fabs(self.z), alpha) * math.exp(-math.pow(math.log(1/(1-probabilities[i])), gamma))))
            else:
                values.append((math.pow(self.c+self.z, alpha) * math.exp(-math.pow(math.log(1/probabilities[i]), gamma)))-(math.pow(math.fabs(self.z + self.e), alpha) * math.exp(-math.pow(math.log(1/(1-probabilities[i])), gamma))))
        return values

    #--Deprecated--
    def bestAlternative(self,values,action):
        highest = max(values)
        if highest!=action:
            return highest
        temp = list(filter(lambda a: a != highest, values))
        if len(temp)==0:
            return -100
        return max(temp)

#     Compute P_I for CPT models --Deprecated--
    def computePI(self, values, actionTaken,lossType):
        z = self.bestAlternative(values,values[actionTaken])
        if (z==-100):
            if lossType=="logit":
                return 0.25
            else:
                return 0
        z = values[actionTaken]-z
        if lossType=="softmax":
            return z
        return 1/(1+math.exp(-z))


    #action in 0,...,numAgents
    def computeLoss(self,params,probabilities,chosen_action,lossType,loss_function,modelName):
        current_models = ["nb","nb-pt","cent","cent-pt"]
        if (modelName not in current_models):
            assert(False)

        prospects= []
        for probability in probabilities[0:self.teamSize]:
            prospectSuccess = self.c, probability
            prospectFailure = self.z, 1-probability
            prospects.append((prospectSuccess,prospectFailure))

        # print("prospects:")
        # print(prospects)

        cpt_vals_option = evaluateProspectVals(params,prospects)

        prospects= []
        for probability in probabilities[self.teamSize:]:
            prospectSuccess = self.c +self.e, probability
            prospectFailure = self.z +self.e, 1-probability
            prospects.append((prospectSuccess,prospectFailure))

        # print("probabilities: ")
        # print(probabilities)
        cpt_vals_agent = evaluateProspectVals(params,prospects)

        cpt_vals = []
        for r in cpt_vals_option:
            cpt_vals.append(r)

        for r in cpt_vals_agent:
            cpt_vals.append(r)

        
        soft_prob = self.softmax(cpt_vals)
        # print(soft_prob)
        predicted = np.argmax(soft_prob)
        ground = chosen_action
        # print("predicted: ")
        # print(predicted)
        # print("ground: ")
        # print(chosen_action)
        acc = 0
        if predicted == ground:
            acc = 1

        loss = 0
        if loss_function == "binary":
            arg = soft_prob[chosen_action]
            if np.isnan(arg):
                arg = 1.
            if arg == 0:
                arg += 0.0000001
            loss = -1. * math.log(arg)

        elif loss_function == "variational_H_qp":

            i = np.argmax(soft_prob)
            j = chosen_action
            # print(soft_prob)

            if i == j:
                for k in range(8):
                    arg = soft_prob[k]
                    loss += -arg * math.log(arg)
            else:
                # print("predicted:")
                # print(i)
                # print("chosen_action:")
                # print(j)
                index1 = []
                index2 = []
                for k in range(8):
                    if soft_prob[k] >= soft_prob[j]:
                        index1.append(k)
                    else:
                        index2.append(k)
                m = 0
                for ind in index1:
                    m += soft_prob[ind]
                
                variation_q =  np.zeros(8)
                for ind in index1:
                    variation_q[ind] = 1.0 * m / len(index1)

                for ind in index2:
                    variation_q[ind] = soft_prob[ind]



                # variation_q =  np.zeros(8)
                # alpha_star = 0.5 * (soft_prob[i] + soft_prob[j])
                # variation_q[i] = alpha_star
                # variation_q[j] = alpha_star
                # index1 = []
                # index2 = []
                # index1.append(i)
                # index1.append(j)
                # for k in range(8):
                #     index2.append(int(k))
                # index3 = list(set(index2) - set(index1))
                
                # max_ind = j
                # for k in index3:
                #     variation_q[k] = soft_prob[k]
                #     if variation_q[k] > alpha_star:
                #         max_ind = k

                # if max_ind != j:
                #     s = 0
                #     for k in index3:
                #         s += variation_q[k]
                #     ratio = 1.0 * variation_q[max_ind] / s
                #     beta = 1.0 * (2 * ratio - 2.0 * alpha_star / (1 - 2 * alpha_star)) / (2 * ratio + 1)
                #     variation_q[i] = 1.0 * ratio / (2 * ratio + 1)
                #     variation_q[j] = 1.0 * ratio / (2 * ratio + 1)
                #     for k in index3:
                #         variation_q[k] *= (1 - beta)
                    # print("sum  of q:")
                    # print(variation_q)
                    # print(np.sum(variation_q))

                # print("soft_prob: ")
                # print(soft_prob)
                # print("variation_q: ")
                # print(variation_q)
                # print(sum(variation_q))
                for k in range(8):
                    loss += -soft_prob[k] * math.log(variation_q[k])
        else:
            i = np.argmax(soft_prob)
            j = chosen_action
            if i == j:
                for k in range(8):
                    arg = soft_prob[k]
                    loss += -arg * math.log(arg)
            else:

                index1 = []
                index2 = []
                for k in range(8):
                    if soft_prob[k] >= soft_prob[j]:
                        index1.append(k)
                    else:
                        index2.append(k)
                
                variation_q =  np.zeros(8)
                for ind in index1:
                    variation_q[ind] = 1.0  / len(index1)


                # # print("predicted:")
                # # print(i)
                # # print("chosem_action:")
                # # print(j)
                # variation_q =  np.zeros(8)
                # index1 = []
                # index2 = []
                # index1.append(i)
                # index1.append(j)
                # for k in range(8):
                #     index2.append(int(k))
                # index3 = list(set(index2) - set(index1))

                # l = np.sqrt(soft_prob[i] * soft_prob[j])
                # index = []
                # for k in index3:
                #     if soft_prob[k] > l:
                #         index.append(k)

                # if len(index) == 0:
                #     variation_q[i] = 0.5
                #     variation_q[j] = 0.5

                # else:
                #     variation_q[i] = 1. / (len(index) + 2)
                #     variation_q[j] = 1. / (len(index) + 2)
                #     for kk in range(len(index)):
                #         variation_q[index[kk]] = 1. / (len(index) + 2)
                # # print("soft_prob: ")
                # # print(soft_prob)
                #print("variation_q: ")
                #print(variation_q)
                #print(sum(variation_q))

                for k in range(8):
                    loss += -variation_q[k] * math.log(soft_prob[k])
                    #if variation_q[k] == 0:
                    #    variation_q[k] = 0.00000001
                    #loss += variation_q[k] * math.log(variation_q[k])


        return loss, acc, predicted, ground


    def computeCPTLoss(self,params,probabilities,lossType,loss_function,modelName):
        total_loss = 0
        per_question_loss = [None for _ in range(self.numQuestions)]
        per_question_acc = [None for _ in range(self.numQuestions)]

        per_question_ground = [None for _ in range(self.numQuestions)]
        per_question_predicted = [None for _ in range(self.numQuestions)]

        length = len(probabilities)


        start = 0
        if length==self.testSetSize:
            start = self.trainingSetSize
        
        q_ind = np.zeros(length)
        for question_num in range(length):
            #Here - How to handle consensus not reached case
            if self.actionTaken[start+question_num]==-1:
                # print("consensus not reached:")
                # print(question_num)
                q_ind[question_num] = 1
                continue

            assert(self.actionTaken[start+question_num] in range(self.teamSize+self.numAgents))
            loss, acc, predicted, ground = self.computeLoss(params,probabilities[question_num],self.actionTaken[start+question_num],lossType,loss_function,modelName)
            per_question_loss[start+question_num] = loss
            per_question_acc[start+question_num] = acc
            per_question_ground[start+question_num] = ground
            per_question_predicted[start+question_num] = predicted

        return per_question_loss, per_question_acc, per_question_ground,per_question_predicted,q_ind


    def computeAverageLossPerTeam(self,params, probabilities, lossType, loss_function, modelName):

        if modelName == "random":

            per_question_loss = [None for _ in range(self.numQuestions)]
            per_question_acc = [None for _ in range(self.numQuestions)]

            length = len(probabilities)
            start = 0
            if length==self.testSetSize:
                start = self.trainingSetSize

            for question_num in range(length):
                #Here - How to handle consensus not reached case
                if self.actionTaken[start+question_num]==-1:
                    continue
                
                assert(self.actionTaken[start+question_num] in range(self.teamSize+self.numAgents))
                prob = 1.0/8
                per_question_loss[start+question_num] = -1.0*math.log(prob)
                per_question_acc[start+question_num] = prob
                ground = [None for _ in range(8)]
                predicted = [None for _ in range(8)]
                q_ind = []

                # per_question_loss, per_question_acc, ground,predicted = self.computeCPTLoss(params,probabilities,lossType,loss_function,'nb')

        else:

            per_question_loss, per_question_acc, ground,predicted,q_ind = self.computeCPTLoss(params,probabilities,lossType,loss_function,modelName)


        total_loss = 0
        all_loss = []
        count = 0
        for loss in per_question_loss:
            if (loss != None):
                all_loss.append(loss)
                total_loss += loss
                count += 1

        if count!=0:
            total_loss /= count

        total_acc = 0
        count_acc = 0
        for acc in per_question_acc:
            if (acc != None):
                total_acc += acc
                count_acc += 1

        if count_acc!=0:
            total_acc /= count_acc

        ground1 = []
        predicted1 = []
        for ele in ground:
            if ele != None:
                ground1.append(ele)

        for ele in predicted:
            if ele != None:
                predicted1.append(ele)

        # print("ground1: ")
        # print(ground1)
        # print("predicted1: ")
        # print(predicted1)

        return all_loss, total_loss, total_acc, ground1,predicted1,q_ind


    def chooseCPTParameters(self, probabilities,lossType,loss_function,modelName):
        # hAlpha, hGamma,hLambda =  (None,None,None)
        Alpha, Beta, Lambda, GammaGain, GammaLoss =  (None,None,None,None,None)

        Loss = np.float("Inf")

        for alpha in np.arange(0,1.1,0.1):
            for lamda in np.arange(1,11,1):
                for gammaGain in np.arange(0,1.1,0.1):
                    for gammaLoss in np.arange(0,1.1,0.1):

                        all_loss, loss_cpt, acc_cpt, ground,predicted,q_ind = self.computeAverageLossPerTeam((alpha,alpha,lamda,gammaGain,gammaLoss),probabilities,lossType,loss_function,modelName)

                        if (loss_cpt<Loss):
                            Loss = loss_cpt
                            Alpha = alpha
                            Beta = alpha
                            Lambda = lamda
                            GammaGain = gammaGain
                            GammaLoss = gammaLoss

        assert(Alpha != None)
        assert(Beta != None)
        assert(Lambda != None)
        assert(GammaGain != None)
        assert(GammaLoss != None)

        return (Alpha, Beta, Lambda, GammaGain, GammaLoss)


    def randomModel(self):
        prAgent = len(self.machineAskedQuestions)/self.numQuestions
        prHuman = 1.0-prAgent
        qi = list()
        for i in range(self.trainingSetSize,self.numQuestions):
            temp = [0.25*prHuman for j in range(0,self.teamSize)]
            for j in range(0,self.teamSize):
                temp.append(0.25*prAgent)
            qi.append(temp)
        return qi

    # Agent i has agent i + teamSize
    def getAgentForHuman(self, k):
        return self.teamSize + k

if __name__ == '__main__':
    directory = "logs/"
#     cleanEventLog(directory+"event_log.csv")
#     insertInfluenceMatrixNumber(directory+"event_log-Copy.csv")
#     addMissingLogs(directory, directory+"event_log.csv")
    testSize = 15
    batchNumbers = [10,11,12,13,17,20,21,28,30,33,34,36,37,38,39,41,42,43,44,45,48,49,74,75,77,82,84,85,87,88]
    RATIONAL_PARAMS= (1,1,1)
    NUM_CPT_PARAMS = 2
    NUM_QUESTIONS = 45

    team = pd.read_csv(directory+"team.csv",sep=',',quotechar="|",names=["id","sessionId","roundId", "taskId"])

    nbLoss = list()
    centLoss = list()
    nbPTLoss = list()
    centPTLoss = list()
    rdLoss = list()

    nbAcc = list()
    centAcc = list()
    nbPTAcc = list()
    centPTAcc = list()
    rdAcc = list()

    Pre = list()
    Re = list()
    F1 = list()

    nbAlpha = list()
    nbGamma = list()
    centAlpha = list()
    centGamma = list()
    
    rdOptionLoss = list()
    rdAgentLoss = list()
    nbOptionLoss = list()
    nbAgentLoss = list()
    centOptionLoss = list()
    centAgentLoss = list()
    nbPTOptionLoss = list()
    nbPTAgentLoss = list()
    centPTOptionLoss = list()
    centPTAgentLoss = list()
    total_group_accuracy = list()
    teamAccuracies = list()
    allBestHumanAccuracy = list()
    allBestMachineAccuracy = list()
    group_accuracy_per_question = list()
    group_accuracy_over_time = np.zeros(NUM_QUESTIONS)

    #lossType = "logit"
    lossType = "softmax"

    # loss_function = "binary"
    # loss_function = "variational_H_qp"
    loss_function = "variational_H_pq"
    
    all_losses = []

    option_results = []
    agent_results = []

    best_parameters = []
    prob_test = []

    all_data = []

    team_rewards_d1 = []
    human_rewards_d1 = []
    agent_rewards_d1 = []

    team_rewards_d2 = []
    human_rewards_d2 = []
    agent_rewards_d2 = []

    team_rewards_d2_correct = []
    human_rewards_d2_correct = []
    agent_rewards_d2_correct = []

    team_rewards_d2_wrong = []
    human_rewards_d2_wrong = []
    agent_rewards_d2_wrong = []

    num_not_concensus_d1 = []
    num_not_concensus_d2 = []
    num_d1 = []
    num_d2 = []

    num_agent_correct_response = []
    num_agent_wrong_response = []
    num_agent_correct_response_noncon = []
    num_agent_wrong_response_noncon = []


    consulted_agent = []
    cnt_a1 = 0
    cnt_a2 = 0
    cnt_a3 = 0
    cnt_a4 = 0
    cnt_a = 0
    agent_answer = []
    cnt_r = 0
    cnt_w = 0

    agent_acc_dt2 = []
    human_acc_dt2 = []
    group_acc_dt2 = []
    non_consensus_dt2 = []
    ai1 = 0
    ai2 = 0
    ai3 = 0
    ai4 = 0

    if_equal = 0
    for i in range(len(team)):   
        
        if team.iloc[i]['id'] in batchNumbers:
            team_data = {}
            print("i: ")
            print(i)
            print("Values of team", team.iloc[i]['id'])
            # if i == 51:
            #     continue

            #Create model
            model = HumanDecisionModels(team.iloc[i]['id'], directory)

            expectedPerformance, rewardsNB1, rewardsCENT1,probabilitiesNB1,probabilitiesCENT1, probabilitiesRANDOM, group_accuracy_per_question, best_human_accuracy, best_machine_accuracy, all_accuracy,human_p,before_consensus, agent_acc_for_dt2,human_acc_for_dt2,group_acc_for_dt2,non_consensus_for_dt2, eig_cen = model.calculateRewards()
            # print("rewardsNB1: ")
            # print(rewardsNB1)
            # print("probabilitiesNB1: ")
            # print(probabilitiesNB1)
            # print("expectedPerformance:")
            # print(np.asarray(expectedPerformance[4:]))

            team_reward = 0
            team_agent_acc = np.asarray(all_accuracy)
            num_corr = 0
            for i1 in range(0,team_agent_acc.shape[1]):
                if team_agent_acc[0][i1] == 1:
                    if team_agent_acc[1][i1] == 0:
                        team_reward += -2
                    else:
                        team_reward += 3
                        num_corr+=1
                else:
                    if team_agent_acc[1][i1] == 0:
                        team_reward += -1
                    else:
                        team_reward += 4
            # print("number of correct choices:")
            # print(num_corr)

            teamAccuracies.append(team_reward)
            bhuman_reward = best_human_accuracy * 45 * 4 - (1-best_human_accuracy) * 45
            allBestHumanAccuracy.append(bhuman_reward)
            # if bhuman_reward >= team_reward:
            #     print("This team cannot defeat the best human:")
            #     print(i)

            # total_group_accuracy = sum(group_accuracy_per_question)/len(group_accuracy_per_question) #Accuracy of one team over all questions
            # teamAccuracies.append(total_group_accuracy)  #total accuracy of all teams
            # allBestHumanAccuracy.append(best_human_accuracy) #Best human accuracy for all teams
            allBestMachineAccuracy.append(best_machine_accuracy)
            group_accuracy_over_time = [(group_accuracy_over_time[j-1] + sum(group_accuracy_per_question[:j])/(j)) for j in range(1,len(group_accuracy_per_question)+1)]

            # print("action: ")
            # print(model.actionTaken) 

            # all_loss, loss, acc, ground,predicted, q_ind_not_con = model.computeAverageLossPerTeam(RATIONAL_PARAMS,probabilitiesRANDOM[model.trainingSetSize:],lossType,loss_function,"random")
            # for total_loss in all_loss:
            #     all_losses.append(total_loss)   
            # # loss = model.computeAverageLossPerTeam(expectedPerformance,probabilitiesNB1,lossType,loss_function,"random")
            # rdLoss.append(loss)
            # # Compute losses for NB and CENT
            # # print(RATIONAL_PARAMS)
            # rdAcc.append(acc)


            # all_loss, loss, acc, ground,predicted, q_ind_not_con  = model.computeAverageLossPerTeam(RATIONAL_PARAMS,probabilitiesNB1[model.trainingSetSize:],lossType,loss_function,"nb")
            # for total_loss in all_loss:
            #     all_losses.append(total_loss) 
            # # loss = model.computeAverageLossPerTeam(RATIONAL_PARAMS,probabilitiesNB1,lossType,loss_function,"nb")
            # # print(RATIONAL_PARAMS)
            # nbLoss.append(loss)
            # nbAcc.append(acc)
            # # print(probabilitiesNB1[model.trainingSetSize:])
            # # print(probabilitiesNB1)
            # # continue
            # prob_test.append(probabilitiesNB1[model.trainingSetSize:])#4::5

            # all_loss, loss, acc, ground,predicted,q_ind_not_consensus = model.computeAverageLossPerTeam(RATIONAL_PARAMS,probabilitiesCENT1[model.trainingSetSize:],lossType,loss_function,"cent")
            # for total_loss in all_loss:
            #     all_losses.append(total_loss) 
            # # loss, optionLoss, agentLoss = model.computeAverageLossPerTeam(RATIONAL_PARAMS,probabilitiesCENT1,lossType,loss_function,"cent")
            # centLoss.append(loss)
            # centAcc.append(acc)
            # prob_test.append(probabilitiesCENT1[model.trainingSetSize:])
            
            # ai = []
            # for ele in ground:
            #     if ele >3:
            #         ai.append(ele - 4)
            #         if ele -4 == 0:
            #             ai1 += 1
            #         elif ele -4 == 1:
            #             ai2 += 1
            #         elif ele -4 == 2:
            #             ai3 += 1
            #         elif ele -4 == 3:
            #             ai4 += 1

            # print("agent consulted order:")
            # print(ai)
            # plf.plotAgentUse(ai, i)



            b_param = []
            # Train alpha,gammma losses for NB-PT
            # hAlpha,hGamma,hLambda = model.chooseCPTParameters(probabilitiesNB1,lossType,loss_function,"nb-pt")
            Alpha, Beta, Lambda, GammaGain, GammaLoss = model.chooseCPTParameters(probabilitiesNB1[:model.trainingSetSize],lossType,loss_function,"nb-pt")
            # print("PT-NB",Alpha, Beta, Lambda, GammaGain, GammaLoss)
            b_param.append(round(Alpha,1))
            b_param.append(round(Beta,1))
            b_param.append(round(Lambda,1))
            b_param.append(round(GammaGain,1))
            b_param.append(round(GammaLoss,1))
            print("PT-NB:")
            print(b_param)

            all_loss, loss, acc, ground,predicted, q_ind_not_con = model.computeAverageLossPerTeam((Alpha, Beta, Lambda, GammaGain, GammaLoss),probabilitiesNB1[model.trainingSetSize:],lossType,loss_function,"nb-pt")
            for total_loss in all_loss:
                all_losses.append(total_loss) 
            # loss = model.computeAverageLossPerTeam((hAlpha,hGamma,hLambda),probabilitiesNB1,lossType,loss_function,"nb-pt")
            nbPTLoss.append(loss)
            nbPTAcc.append(acc)
            best_parameters.append(b_param)


            b_param = []
            # Train alpha,gammma losses for CENT-PT
            # hAlpha,hGamma,hLambda = model.chooseCPTParameters(probabilitiesCENT1,lossType,loss_function,"cent-pt")
            Alpha, Beta, Lambda, GammaGain, GammaLoss = model.chooseCPTParameters(probabilitiesCENT1[:model.trainingSetSize],lossType,loss_function,"cent-pt")
            # print("CENT-PT",Alpha, Beta, Lambda, GammaGain, GammaLoss)
            b_param.append(round(Alpha,1))
            b_param.append(round(Beta,1))
            b_param.append(round(Lambda,1))
            b_param.append(round(GammaGain,1))
            b_param.append(round(GammaLoss,1))
            print("PT-CENT:")
            print(b_param)
            
            all_loss, loss, acc, ground,predicted, q_ind_not_con = model.computeAverageLossPerTeam((Alpha, Beta, Lambda, GammaGain, GammaLoss),probabilitiesCENT1[model.trainingSetSize:],lossType,loss_function,"cent-pt")
            for total_loss in all_loss:
                all_losses.append(total_loss) 
            # loss = model.computeAverageLossPerTeam((hAlpha,hGamma,hLambda),probabilitiesCENT1,lossType,loss_function,"cent-pt")
            centPTLoss.append(loss)
            centPTAcc.append(acc)
            best_parameters.append(b_param)

            all_predicted = []
            all_ground = []
            # print(len(ground))

            # for ele in consult_agent_id:
            #     consulted_agent.append(ele)
            #     cnt_a += 1
            #     if ele == 0:
            #         cnt_a1 += 1

            #     elif ele == 1:
            #         cnt_a2 += 1

            #     elif ele == 2:
            #         cnt_a3 += 1

            #     else:
            #         cnt_a4 += 1

            
            # for ele in agent_perform:
            #     agent_answer.append(ele)
            #     cnt_a += 1
            #     if ele == 0:
            #         cnt_w += 1

            #     elif ele == 1:
            #         cnt_r += 1


            for ele in agent_acc_for_dt2:
                agent_acc_dt2.append(ele)

            for ele in human_acc_for_dt2:
                human_acc_dt2.append(ele)

            for ele in group_acc_for_dt2:
                group_acc_dt2.append(ele)

            for ele in non_consensus_for_dt2:
                non_consensus_dt2.append(ele)


            # for ele in predicted:
            #     all_predicted.append(ele)

            # print("all_ground:")
            # print(all_ground)
            
            # print("consult_agent:")#
            # print(team_agent_acc[0][:])#
            # print("team acc:")#
            # print(team_agent_acc[1][:])#
            # print("agent response:")#
            # print(team_agent_acc[2][:])#
            # print("consensus not reached:")#
            # print(q_ind_not_consensus)#
            
            # team_reward_d1=0#
            # team_reward_d2=0#
            # best_human_reward_d1=0#
            # best_human_reward_d2=0#
            # avg_agent_reward_d1=0#
            # avg_agent_reward_d2=0#


            # team_perform_d1 = []#
            # team_perform_d2 = []#
            # best_human_perform_d1 = []#
            # best_human_perform_d2 = []#
            # for i1 in range(len(q_ind_not_consensus)):#
            #     if team_agent_acc[0][i1] == 0 and q_ind_not_consensus[i1] == 0:#
            #         team_perform_d1.append(i1)#

            #     if team_agent_acc[0][i1] == 0:#
            #         best_human_perform_d1.append(i1)#

            #     if team_agent_acc[0][i1] == 1 and q_ind_not_consensus[i1] == 0:#
            #         team_perform_d2.append(i1)#

            #     if team_agent_acc[0][i1] == 1:#
            #         best_human_perform_d2.append(i1)#
            
            # for i1 in team_perform_d1:#
            #     if team_agent_acc[1][i1] == 1:#
            #         team_reward_d1 += 4#
            #     else:#
            #         team_reward_d1 += -1#
            # #for i1 in team_perform_d2:
            # #    team_reward_d1 += -1
            
            # for i1 in team_perform_d2:#
            #     if team_agent_acc[1][i1] == 1:#
            #         team_reward_d2 += 3#
            #     else:#
            #         team_reward_d2 += -2#

            # avg_agent_reward_d1=len(best_human_perform_d1) * 0.75 * 3 - len(best_human_perform_d1) * 0.25 * 2#

            # #for i1 in best_human_perform_d2:
            # #    avg_agent_reward_d1 += -1

            # avg_agent_reward_d2=len(best_human_perform_d2) * 0.75 * 3 - len(best_human_perform_d2) * 0.25 * 2#
            
            # team_reward_d2_correct = 0
            # team_reward_d2_wrong = 0
            
            # for i1 in team_perform_d2:
            #     if team_agent_acc[2][i1] == 1:
            #         if team_agent_acc[1][i1] == 1:#
            #             team_reward_d2_correct += 3
            #         else:
            #             team_reward_d2_correct += -2
            #     else:
            #         if team_agent_acc[1][i1] == 1:#
            #             team_reward_d2_wrong += 3
            #         else:
            #             team_reward_d2_wrong += -2
            
            # cnt_correct = 0
            # cnt_correct_noncon = 0
            # cnt_wrong_noncon = 0
            # for i1 in best_human_perform_d2:
            #     if team_agent_acc[2][i1] == 1:
            #         cnt_correct += 1
            #         if q_ind_not_consensus[i1] == 1:
            #             cnt_correct_noncon += 1
            #     else:
            #         if q_ind_not_consensus[i1] == 1:
            #             cnt_wrong_noncon += 1


            # num_agent_correct_response.append(cnt_correct)
            # num_agent_correct_response_noncon.append(cnt_correct_noncon)
            # num_agent_wrong_response.append(len(best_human_perform_d2)-cnt_correct)
            # num_agent_wrong_response_noncon.append(cnt_wrong_noncon)

            # avg_agent_reward_d2_correct = cnt_correct * 3#
            # avg_agent_reward_d2_wrong = - (len(best_human_perform_d2)-cnt_correct) * 2#


            # h1=human_p[0]#
            # h2=human_p[1]#
            # h3=human_p[2]#
            # h4=human_p[3]#

            # # print(len(h1))
            # # print(len(h2))
            # # print(len(h3))
            # # print(len(h4))
            
            # for i1 in range(len(eig_cen)):
            #     # print((i1+1)*5)
            #     p_h = [sum(h1[:((i1+1)*5)])/((i1+1)*5.0), sum(h2[:((i1+1)*5)])/((i1+1)*5.0), sum(h3[:((i1+1)*5)])/((i1+1)*5.0), sum(h4[:((i1+1)*5)])/((i1+1)*5.0)]
            #     m_p_h = max(p_h)
            #     b_ind = [i2 for i2, j2 in enumerate(p_h) if j2 == m_p_h]

            #     eigen_centrality = eig_cen[i1]
            #     m_eigen_centrality = max(eigen_centrality)
            #     e_ind = [i2 for i2, j2 in enumerate(eigen_centrality) if j2 == m_eigen_centrality]

            #     if list(set(b_ind) & set(e_ind)):
            #         if_equal += 1



            # h1_1=[]#
            # h2_1=[]#
            # h3_1=[]#
            # h4_1=[]#

            # h1_2=[]#
            # h2_2=[]#
            # h3_2=[]#
            # h4_2=[]#

            # for i1 in best_human_perform_d1:#
            #     h1_1.append(h1[i1])#
            #     h2_1.append(h2[i1])#
            #     h3_1.append(h3[i1])#
            #     h4_1.append(h4[i1])#

            # for i1 in best_human_perform_d2:#
            #     h1_2.append(h1[i1])#
            #     h2_2.append(h2[i1])#
            #     h3_2.append(h3[i1])#
            #     h4_2.append(h4[i1])#

            # best_human_accuracy_d1 = max([sum(h1_1)/len(h1_1), sum(h2_1)/len(h2_1), sum(h3_1)/len(h3_1), sum(h4_1)/len(h4_1)])#
            # best_human_accuracy_d2 = max([sum(h1_2)/len(h1_2), sum(h2_2)/len(h2_2), sum(h3_2)/len(h3_2), sum(h4_2)/len(h4_2)])#

            # best_human_reward_d1 = best_human_accuracy_d1 * len(best_human_perform_d1) * 4 - (1 - best_human_accuracy_d1) * len(best_human_perform_d1)#
            # best_human_reward_d2 = best_human_accuracy_d2 * len(best_human_perform_d2) * 3 - (1 - best_human_accuracy_d2) * len(best_human_perform_d2) * 2#

            # best_human_reward_d2_correct = best_human_accuracy_d2 * cnt_correct * 3 - (1 - best_human_accuracy_d2) * cnt_correct * 2#
            # best_human_reward_d2_wrong = best_human_accuracy_d2 * (len(best_human_perform_d2)-cnt_correct) * 3 - (1 - best_human_accuracy_d2) * (len(best_human_perform_d2)-cnt_correct) * 2#

            # team_rewards_d1.append(team_reward_d1)#
            # human_rewards_d1.append(best_human_reward_d1)#
            # agent_rewards_d1.append(avg_agent_reward_d1)#

            # team_rewards_d2.append(team_reward_d2)#
            # human_rewards_d2.append(best_human_reward_d2)#
            # agent_rewards_d2.append(avg_agent_reward_d2)#

            # team_rewards_d2_correct.append(team_reward_d2_correct)#
            # human_rewards_d2_correct.append(best_human_reward_d2_correct)#
            # agent_rewards_d2_correct.append(avg_agent_reward_d2_correct)#

            # team_rewards_d2_wrong.append(team_reward_d2_wrong)#
            # human_rewards_d2_wrong.append(best_human_reward_d2_wrong)#
            # agent_rewards_d2_wrong.append(avg_agent_reward_d2_wrong)#

            # team_data["human_performance"]=human_p#
            # team_data["consult_agent"]=list(team_agent_acc[0][:])#
            # team_data["team_performance"]=list(team_agent_acc[1][:])#
            # team_data["no_consensus"]=q_ind_not_consensus#

            # all_data.append(team_data)#


    #         result_tuple=[]#

    #         no_con_d1 = 0#
    #         no_con_d2 = 0#
    #         num_in_d1 = 0#
    #         num_in_d2 = 0#



    #         for i1 in range(9):#

    #             cnt_before_consensus=0#
    #             cnt_consult_agent=0#
    #             cnt_option=5#
    #             cnt_after_consensus=0#
    #             cnt_agent_not_consensus=0#
    #             for j1 in range(i1*5, i1*5+5):#
    #                 if before_consensus[j1]==1:#
    #                     cnt_before_consensus += 1#

    #             for j1 in range(i1*5, i1*5+5):#
    #                 if team_agent_acc[0][j1] == 1:#
    #                     cnt_consult_agent += 1#
    #                     if q_ind_not_consensus[j1] == 1:#
    #                         cnt_agent_not_consensus += 1#

    #                 if team_agent_acc[0][j1] == 0 and q_ind_not_consensus[j1] == 1:#
    #                     cnt_after_consensus += 1#
                     
    #             cnt_option -= cnt_consult_agent#
    #             cnt_option -= cnt_after_consensus#

                
    #             result_tuple.append((cnt_before_consensus,cnt_option,cnt_consult_agent,cnt_after_consensus,cnt_consult_agent-cnt_agent_not_consensus,cnt_agent_not_consensus))#
    #             no_con_d1 += cnt_after_consensus#
    #             no_con_d2 += cnt_agent_not_consensus#
                
    #             num_in_d2 += cnt_consult_agent#

    #         num_not_concensus_d1.append(no_con_d1)#
    #         num_not_concensus_d2.append(no_con_d2)#
    #         num_d2.append(num_in_d2)#
            
    #         print("results:")#
    #         print(result_tuple)#

    # print("no consensus d1:")#
    # print(num_not_concensus_d1)#
    # print("no consensus d2:")#
    # print(num_not_concensus_d2)#
    # print("num in d2:")#
    # print(num_d2)#


            # print("all_predicted")
            # print(all_predicted)

            # count_option = 0
            # true_option = 0
            # count_agent = 0
            # true_agent = 0
            # option_r = []
            # agent_r = []
            # for ind in range(len(all_predicted)):
            #     if all_ground[ind] < 4:
            #         count_option += 1
            #         if all_predicted[ind] == all_ground[ind]:
            #             true_option += 1
            #     else:
            #         count_agent += 1
            #         if all_predicted[ind] == all_ground[ind]:
            #             true_agent += 1

            # option_r.append(count_option)
            # option_r.append(true_option)
            # option_r.append(1.0 * true_option / count_option)
            # if count_agent != 0:
            #     agent_r.append(count_agent)
            #     agent_r.append(true_agent)
            #     agent_r.append(1.0 * true_agent / count_agent)
            #     agent_results.append(agent_r)

            # option_results.append(option_r)
            
            # result = precision_recall_fscore_support(np.asarray(all_ground), np.asarray(all_predicted), average='macro')
            
            # if not np.isnan(result[0]):
            #     Pre.append(result[0])

            # if not np.isnan(result[1]):    
            #     Re.append(result[1])

            # if not np.isnan(result[2]):
            #     F1.append(result[2])   

    #with open('./pt_cent_param_l2.data', 'wb') as filehandle:
    #    # store the data as binary data stream
    #    pickle.dump(best_parameters, filehandle)  

    # with open('./probability_cent_for_pt.data', 'wb') as filehandle:
    #     # store the data as binary data stream
    #     pickle.dump(prob_test, filehandle)     

    # with open('./all_data.data', 'wb') as filehandle:#
    #     # store the data as binary data stream
    #     pickle.dump(all_data, filehandle) #


    # group_accuracy_over_time = [(group_accuracy_over_time[i]/(len(teamAccuracies))) for i in range(len(group_accuracy_over_time))]
    # ratio1 = [teamAccuracies[i]/allBestHumanAccuracy[i] for i in range(len(teamAccuracies))]
    # ratio2 = [teamAccuracies[i]/allBestMachineAccuracy[i] for i in range(len(teamAccuracies))]
    # ratio3 = [allBestHumanAccuracy[i]/allBestMachineAccuracy[i] for i in range(len(teamAccuracies))]

    # plf.groupVSbestHumanAccuracy(ratio3)

    print("team accuracy:")#
    print(teamAccuracies)#
    print("allBestHumanAccuracy:")#
    print(allBestHumanAccuracy)#

    print("team_rewards_d1")#
    print(team_rewards_d1)#
    print("human_rewards_d1")#
    print(human_rewards_d1)#
    print("agent_rewards_d1")#
    print(agent_rewards_d1)#

    print("team_rewards_d2")#
    print(team_rewards_d2)#
    print("human_rewards_d2")#
    print(human_rewards_d2)#
    print("agent_rewards_d2")#
    print(agent_rewards_d2)#

    print("team_rewards_d2_correct")#
    print(team_rewards_d2_correct)#
    print("human_rewards_d2_correct")#
    print(human_rewards_d2_correct)#
    print("agent_rewards_d2_correct")#
    print(agent_rewards_d2_correct)#

    print("team_rewards_d2_wrong")#
    print(team_rewards_d2_wrong)#
    print("human_rewards_d2_wrong")#
    print(human_rewards_d2_wrong)#
    print("agent_rewards_d2_wrong")#
    print(agent_rewards_d2_wrong)#

    print("num_agent_correct_response:")
    print(num_agent_correct_response)
    print("num_agent_wrong_response:")
    print(num_agent_wrong_response)

    print("num_agent_correct_response_non_consensus:")
    print(num_agent_correct_response_noncon)
    print("num_agent_wrong_response_non_consensus:")
    print(num_agent_wrong_response_noncon)

    print(np.sum(num_agent_correct_response_noncon))
    print(np.sum(num_agent_correct_response))
    print(np.sum(num_agent_correct_response_noncon)/np.sum(num_agent_correct_response))
    print(np.sum(num_agent_wrong_response_noncon))
    print(np.sum(num_agent_wrong_response))
    print(np.sum(num_agent_wrong_response_noncon)/np.sum(num_agent_wrong_response))
    # print("option_results: ")
    # print(option_results)
    # print(np.mean(np.asarray(option_results), axis=0))
    # print(np.std(np.asarray(option_results), axis=0))
    # print("agent_results: ")
    # print(agent_results)
    # print(np.mean(np.asarray(agent_results), axis=0))
    # print(np.std(np.asarray(agent_results), axis=0))


    # print("all_losses:")
    # print(all_losses)

    # print("precision: ",np.mean(Pre),np.std(Pre))
    # print("recall: ",np.mean(Re),np.std(Re))
    # print("f1: ",np.mean(F1),np.std(F1))

    # print("all_ground: ")
    # print(all_ground)
    # print("all_predicted: ")
    # print(all_predicted)

    # plf.histPlot(nbLoss, "nb")
    # plf.histPlot(centLoss, "cent")
    # plf.histPlot(nbPTLoss, "nb-pt")
    # plf.histPlot(centPTLoss, "cent-pt")
    # plf.histPlot(rdLoss, "random")

    print("rd loss:")
    print(rdLoss)
    print("nb loss:")
    print(nbLoss)
    # print(stats.ttest_ind(np.asarray(rdLoss),np.asarray(nbLoss)))
    # print(stats.mannwhitneyu(np.asarray(rdLoss),np.asarray(nbLoss)))
    print("cent loss:")
    print(centLoss)
    # print(stats.ttest_ind(np.asarray(rdLoss),np.asarray(centLoss)))
    # print(stats.mannwhitneyu(np.asarray(rdLoss),np.asarray(centLoss)))
    print("nbPT loss:")
    print(nbPTLoss)
    # print(stats.ttest_ind(np.asarray(rdLoss),np.asarray(nbPTLoss)))
    # print(stats.mannwhitneyu(np.asarray(rdLoss),np.asarray(nbPTLoss)))
    print("centPT loss:")
    print(centPTLoss)
    # print(stats.ttest_ind(np.asarray(rdLoss),np.asarray(centPTLoss)))
    # print(stats.mannwhitneyu(np.asarray(rdLoss),np.asarray(centPTLoss)))

    # print(stats.mannwhitneyu(np.asarray(nbLoss),np.asarray(nbPTLoss)))
    # print(stats.mannwhitneyu(np.asarray(centLoss),np.asarray(centPTLoss)))
    # print(stats.mannwhitneyu(np.asarray(nbLoss),np.asarray(centPTLoss)))
    # print(stats.mannwhitneyu(np.asarray(centLoss),np.asarray(nbPTLoss)))

    # print("NB1 acc:")
    # print(nbAcc)
    # print("CENT1 acc:")
    # print(centAcc)
    # print("PT-NB-1 acc:")
    # print(nbPTAcc)
    # print("PT-CENT-1 acc:")
    # print(centPTAcc)
    # print("Random acc:")
    # print(rdAcc)

    # print("NB1 acc:",np.mean(nbAcc),np.std(nbAcc))
    # print("CENT1 acc:",np.mean(centAcc),np.std(centAcc))
    # print("PT-NB-1 acc:", np.mean(nbPTAcc),np.std(nbPTAcc))
    # print("PT-CENT-1 acc:",np.mean(centPTAcc),np.std(centPTAcc))
    # print("Random acc:",np.mean(rdAcc),np.std(rdAcc))

    # print("consulted_agent:")
    # print(consulted_agent)
    # print("cnt_a")
    # print(cnt_a)
    # print("cnt_a1")
    # print(cnt_a1)
    # print("cnt_a2")
    # print(cnt_a2)
    # print("cnt_a3")
    # print(cnt_a3)
    # print("cnt_a4")
    # print(cnt_a4)

    # print("consulted_agent:")
    # print(agent_answer)
    # print("cnt_a")
    # print(cnt_a)
    # print("cnt_wrong")
    # print(cnt_w)
    # print("cnt_right")
    # print(cnt_r)

    # print(agent_acc_dt2)
    # print(human_acc_dt2)
    # print(group_acc_dt2)
    # print(non_consensus_dt2)

    print("if equal:")
    print(if_equal)

    print("ai1:")
    print(ai1)
    print("ai2:")
    print(ai2)
    print("ai3:")
    print(ai3)
    print("ai4:")
    print(ai4)

    c1=0
    c2=0
    c3=0
    c4=0

    print("agent_acc_dt2 length:")
    print(len(agent_acc_dt2))
    print(sum(agent_acc_dt2))

    for index in range(len(group_acc_dt2)):
        if group_acc_dt2[index] == 0:
            c1+=1
            if len(set(human_acc_dt2[index])) == 1:
                c4+=0
            if agent_acc_dt2[index] == 1:
                c3+=1

        if non_consensus_dt2[index] == -1 and group_acc_dt2[index] == 0:
            c2+=1
            
        # if non_consensus_dt2[index] == -1 and agent_acc_dt2[index] == 1:
        #     c3+=1

    print(c1)
    print(c2)
    print(c3)
    print(c4)
    



    print("NB1 ",np.mean(nbLoss),np.std(nbLoss))
    print("CENT1 ",np.mean(centLoss),np.std(centLoss))
    print("PT-NB-1 ", np.mean(nbPTLoss),np.std(nbPTLoss))
    print("PT-CENT-1 ",np.mean(centPTLoss),np.std(centPTLoss))
    print("Random ",np.mean(rdLoss),np.std(rdLoss))
