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

        # self.alphas = [1,1,1,1,10,10,10,10]
        # self.betas = np.ones(8, dtype = int)
        # for i in range(4,8):
        #     self.betas[i] = np.random.random_integers(10)

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
        graph = nx.DiGraph()
        for i in range(0,self.teamSize):
            for j in range(0,self.teamSize):
                graph.add_edge(i,j,weight=self.memberInfluences[influenceMatrixIndex][i][j]/100)

        human_centralities = nx.eigenvector_centrality(graph, weight="weight")

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

    def calculatePerformanceProbability(self, questionNumber, idx):
        probabilities = list()
        probabilities = [0 for _ in range(self.teamSize + self.numAgents)]

        for i in range(0,self.teamSize):
            individulResponse = self.lastIndividualResponsesbyQNo[(self.lastIndividualResponsesbyQNo["questionNumber"] == questionNumber) & (self.lastIndividualResponsesbyQNo["sender"] == self.teamMember.iloc[i])]["stringValue"].any()
            print(individulResponse)
            index = self.options[idx].index(individulResponse)
            probabilities[index] += self.centralities[idx][i]
        
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

        return probabilities

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
        h1 = list()
        h2 = list()
        h3 = list()
        h4 = list()
        a1 = list()
        a2 = list()
        a3 = list()
        a4 = list()
        random_p = [0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125]
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

            probabilitiesNB1.append(all_probabilities)
            probabilitiesRANDOM.append(random_p)

            rewardsNB1.append(self.calculateExpectedReward(all_probabilities))
            expectedP.append(expectedPerformance)

        #Compute Reward for CENT1 model
        rewardsCENT1 = list()
        probabilitiesCENT1 = list()
        for i in range(0,self.numCentralityReports):
            self.updateCentrality(self.influenceMatrixIndex)
        for i in range(0,self.numQuestions):
            probabilities = self.calculatePerformanceProbability(self.questionNumbers[i],i)

            # print(probabilities)
            probabilitiesCENT1.append(probabilities)
            rewardsCENT1.append(self.calculateExpectedReward(probabilities))
            # print(rewardsCENT1)

        best_human_accuracy = max([sum(h1)/len(h1), sum(h2)/len(h2), sum(h3)/len(h3), sum(h4)/len(h4)])

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

        return expectedP, rewardsNB1,rewardsCENT1, probabilitiesNB1,probabilitiesCENT1, probabilitiesRANDOM, group_accuracy_per_question, best_human_accuracy, best_machine_accuracy

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

            if i == j:
                for k in range(8):
                    arg = soft_prob[k]
                    loss += -arg * math.log(arg)
            else:
                # print("predicted:")
                # print(i)
                # print("chosen_action:")
                # print(j)
                variation_q =  np.zeros(8)
                alpha_star = 0.5 * (soft_prob[i] + soft_prob[j]) / (soft_prob[i] + soft_prob[j] + (1 - soft_prob[i] - soft_prob[j]) ** 2)
                variation_q[i] = alpha_star
                variation_q[j] = alpha_star
                index1 = []
                index2 = []
                index1.append(i)
                index1.append(j)
                for k in range(8):
                    index2.append(int(k))
                index3 = list(set(index2) - set(index1))

                for k in index3:
                    variation_q[k] = (1 - 2 * alpha_star) * soft_prob[k] / (1- soft_prob[i] - soft_prob[j])

                # print("soft_prob: ")
                # print(soft_prob)
                # print("variation_q: ")
                # print(variation_q)
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
                # print("predicted:")
                # print(i)
                # print("chosem_action:")
                # print(j)
                variation_q =  np.zeros(8)
                index1 = []
                index2 = []
                index1.append(i)
                index1.append(j)
                for k in range(8):
                    index2.append(int(k))
                index3 = list(set(index2) - set(index1))

                l = np.sqrt(soft_prob[i] * soft_prob[j])
                index = []
                for k in index3:
                    if soft_prob[k] > l:
                        index.append(k)

                if len(index) == 0:
                    variation_q[i] = 0.5
                    variation_q[j] = 0.5

                else:
                    variation_q[i] = 1. / (len(index) + 2)
                    variation_q[j] = 1. / (len(index) + 2)
                    for kk in range(len(index)):
                        variation_q[index[kk]] = 1. / (len(index) + 2)
                # print("soft_prob: ")
                # print(soft_prob)
                # print("variation_q: ")
                # print(variation_q)

                for k in range(8):
                    loss += -variation_q[k] * math.log(soft_prob[k])


        return loss, acc, predicted, ground, soft_prob


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
        
        all_soft_prob = []
        for question_num in range(length):
            
            #Here - How to handle consensus not reached case
            if self.actionTaken[start+question_num]==-1:
                # print("consensus not reached:")
                # print(question_num)
                all_soft_prob.append([])
                continue

            assert(self.actionTaken[start+question_num] in range(self.teamSize+self.numAgents))
            loss, acc, predicted, ground, soft_prob = self.computeLoss(params,probabilities[question_num],self.actionTaken[start+question_num],lossType,loss_function,modelName)
            per_question_loss[start+question_num] = loss
            per_question_acc[start+question_num] = acc
            per_question_ground[start+question_num] = ground
            per_question_predicted[start+question_num] = predicted
            
            soft_prob1 = []
            for ele in soft_prob:
                soft_prob1.append(ele)
            soft_prob1.append(ground)
            all_soft_prob.append(soft_prob1)



        return per_question_loss, per_question_acc, per_question_ground,per_question_predicted, all_soft_prob


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
                # prob = 1.0/8
                # per_question_loss[start+question_num] = -1.0*math.log(prob)
                # per_question_acc[start+question_num] = prob
                # ground = [None for _ in range(8)]
                # predicted = [None for _ in range(8)]

                per_question_loss, per_question_acc, ground,predicted, all_soft_prob = self.computeCPTLoss(params,probabilities,lossType,loss_function,'nb')

        else:
            # print("probabilities")
            # print(len(probabilities))

            per_question_loss, per_question_acc, ground,predicted, all_soft_prob = self.computeCPTLoss(params,probabilities,lossType,loss_function,modelName)

        total_loss = 0
        count = 0
        for loss in per_question_loss:
            if (loss != None):
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

        return total_loss, total_acc, ground1,predicted1, all_soft_prob


    def chooseCPTParameters(self, probabilities,lossType,loss_function,modelName):
        # hAlpha, hGamma,hLambda =  (None,None,None)
        Alpha, Beta, Lambda, GammaGain, GammaLoss =  (None,None,None,None,None)

        Loss = np.float("Inf")

        for alpha in np.arange(0,1.1,0.1):
            for lamda in np.arange(1,11,1):
                for gammaGain in np.arange(0,1.1,0.1):
                    for gammaLoss in np.arange(0,1.1,0.1):

                        loss_cpt, acc_cpt, ground,predicted, all_soft_prob = self.computeAverageLossPerTeam((alpha,alpha,lamda,gammaGain,gammaLoss),probabilities,lossType,loss_function,modelName)

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
    loss_function = "variational_H_qp"
    # loss_function = "variational_H_pq"

    
    all_probabilities = []
    for i in range(len(team)):    
        if team.iloc[i]['id'] in batchNumbers:
            print("i: ")
            print(i)
            print("Values of team", team.iloc[i]['id'])
            # if i == 51:
            #     continue

            #Create model
            model = HumanDecisionModels(team.iloc[i]['id'], directory)

            expectedPerformance, rewardsNB1, rewardsCENT1,probabilitiesNB1,probabilitiesCENT1, probabilitiesRANDOM, group_accuracy_per_question, best_human_accuracy, best_machine_accuracy = model.calculateRewards()
            # print("rewardsNB1: ")
            # print(rewardsNB1)
            # print("probabilitiesNB1: ")
            # print(len(probabilitiesNB1))


            total_group_accuracy = sum(group_accuracy_per_question)/len(group_accuracy_per_question) #Accuracy of one team over all questions
            teamAccuracies.append(total_group_accuracy)  #total accuracy of all teams
            allBestHumanAccuracy.append(best_human_accuracy) #Best human accuracy for all teams
            allBestMachineAccuracy.append(best_machine_accuracy)
            group_accuracy_over_time = [(group_accuracy_over_time[j-1] + sum(group_accuracy_per_question[:j])/(j)) for j in range(1,len(group_accuracy_per_question)+1)]

            # print("action: ")
            # print(model.actionTaken) 

            # loss, acc, ground,predicted = model.computeAverageLossPerTeam(RATIONAL_PARAMS,probabilitiesRANDOM[model.trainingSetSize:],lossType,loss_function,"random")
            # # loss = model.computeAverageLossPerTeam(expectedPerformance,probabilitiesNB1,lossType,loss_function,"random")
            # rdLoss.append(loss)
            # # Compute losses for NB and CENT
            # # print(RATIONAL_PARAMS)
            # rdAcc.append(acc)


            # loss, acc, ground,predicted, all_soft_prob = model.computeAverageLossPerTeam(RATIONAL_PARAMS,probabilitiesNB1[model.trainingSetSize:],lossType,loss_function,"nb")
            # # loss = model.computeAverageLossPerTeam(RATIONAL_PARAMS,probabilitiesNB1,lossType,loss_function,"nb")
            # # print(RATIONAL_PARAMS)
            # nbLoss.append(loss)
            # nbAcc.append(acc)
            # # print(probabilitiesNB1)
            # # continue
            # # print(len(all_soft_prob))
            # # for ele in all_soft_prob:
            # #     all_probabilities.append(ele)
            # all_probabilities.append(all_soft_prob)

            loss, acc, ground,predicted, all_soft_prob = model.computeAverageLossPerTeam(RATIONAL_PARAMS,probabilitiesCENT1[model.trainingSetSize:],lossType,loss_function,"cent")
            # loss, optionLoss, agentLoss = model.computeAverageLossPerTeam(RATIONAL_PARAMS,probabilitiesCENT1,lossType,loss_function,"cent")
            centLoss.append(loss)
            centAcc.append(acc)
            # for ele in all_soft_prob:
            #     all_probabilities.append(ele)
            all_probabilities.append(all_soft_prob)

            
            # # Train alpha,gammma losses for NB-PT
            # # hAlpha,hGamma,hLambda = model.chooseCPTParameters(probabilitiesNB1,lossType,loss_function,"nb-pt")
            # Alpha, Beta, Lambda, GammaGain, GammaLoss = model.chooseCPTParameters(probabilitiesNB1[:model.trainingSetSize],lossType,loss_function,"nb-pt")
            # print("PT-NB",Alpha, Beta, Lambda, GammaGain, GammaLoss)
            # loss, acc, ground,predicted = model.computeAverageLossPerTeam((Alpha, Beta, Lambda, GammaGain, GammaLoss),probabilitiesNB1[model.trainingSetSize:],lossType,loss_function,"nb-pt")
            # # loss = model.computeAverageLossPerTeam((hAlpha,hGamma,hLambda),probabilitiesNB1,lossType,loss_function,"nb-pt")
            # nbPTLoss.append(loss)
            # nbPTAcc.append(acc)

            # # Train alpha,gammma losses for CENT-PT
            # # hAlpha,hGamma,hLambda = model.chooseCPTParameters(probabilitiesCENT1,lossType,loss_function,"cent-pt")
            # Alpha, Beta, Lambda, GammaGain, GammaLoss = model.chooseCPTParameters(probabilitiesCENT1[:model.trainingSetSize],lossType,loss_function,"cent-pt")
            # print("CENT-PT",Alpha, Beta, Lambda, GammaGain, GammaLoss)
            # loss, acc, ground,predicted = model.computeAverageLossPerTeam((Alpha, Beta, Lambda, GammaGain, GammaLoss),probabilitiesCENT1[model.trainingSetSize:],lossType,loss_function,"cent-pt")
            # # loss = model.computeAverageLossPerTeam((hAlpha,hGamma,hLambda),probabilitiesCENT1,lossType,loss_function,"cent-pt")
            # centPTLoss.append(loss)
            # centPTAcc.append(acc)

            # all_predicted = []
            # all_ground = []

            # for ele in ground:
            #     all_ground.append(ele)

            # for ele in predicted:
            #     all_predicted.append(ele)

            # result = precision_recall_fscore_support(np.asarray(all_ground), np.asarray(all_predicted), average='macro')
            
            # if not np.isnan(result[0]):
            #     Pre.append(result[0])

            # if not np.isnan(result[1]):    
            #     Re.append(result[1])

            # if not np.isnan(result[2]):
            #     F1.append(result[2])         

    #with open('./action.data', 'wb') as filehandle:
    #    # store the data as binary data stream
    #    pickle.dump(all_probabilities, filehandle)

    # group_accuracy_over_time = [(group_accuracy_over_time[i]/(len(teamAccuracies))) for i in range(len(group_accuracy_over_time))]
    # ratio1 = [teamAccuracies[i]/allBestHumanAccuracy[i] for i in range(len(teamAccuracies))]
    # ratio2 = [teamAccuracies[i]/allBestMachineAccuracy[i] for i in range(len(teamAccuracies))]
    # ratio3 = [allBestHumanAccuracy[i]/allBestMachineAccuracy[i] for i in range(len(teamAccuracies))]

    # plf.groupVSbestHumanAccuracy(ratio3)

    print("precision: ",np.mean(Pre),np.std(Pre))
    print("recall: ",np.mean(Re),np.std(Re))
    print("f1: ",np.mean(F1),np.std(F1))

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
    print(stats.mannwhitneyu(np.asarray(rdLoss),np.asarray(nbLoss)))
    print("cent loss:")
    print(centLoss)
    # print(stats.ttest_ind(np.asarray(rdLoss),np.asarray(centLoss)))
    print(stats.mannwhitneyu(np.asarray(rdLoss),np.asarray(centLoss)))
    print("nbPT loss:")
    print(nbPTLoss)
    # print(stats.ttest_ind(np.asarray(rdLoss),np.asarray(nbPTLoss)))
    print(stats.mannwhitneyu(np.asarray(rdLoss),np.asarray(nbPTLoss)))
    print("centPT loss:")
    print(centPTLoss)
    # print(stats.ttest_ind(np.asarray(rdLoss),np.asarray(centPTLoss)))
    print(stats.mannwhitneyu(np.asarray(rdLoss),np.asarray(centPTLoss)))

    print(stats.mannwhitneyu(np.asarray(nbLoss),np.asarray(nbPTLoss)))
    print(stats.mannwhitneyu(np.asarray(centLoss),np.asarray(centPTLoss)))
    print(stats.mannwhitneyu(np.asarray(nbLoss),np.asarray(centPTLoss)))
    print(stats.mannwhitneyu(np.asarray(centLoss),np.asarray(nbPTLoss)))

    print("NB1 acc:")
    print(nbAcc)
    print("CENT1 acc:")
    print(centAcc)
    print("PT-NB-1 acc:")
    print(nbPTAcc)
    print("PT-CENT-1 acc:")
    print(centPTAcc)
    print("Random acc:")
    print(rdAcc)

    print("NB1 acc:",np.mean(nbAcc),np.std(nbAcc))
    print("CENT1 acc:",np.mean(centAcc),np.std(centAcc))
    print("PT-NB-1 acc:", np.mean(nbPTAcc),np.std(nbPTAcc))
    print("PT-CENT-1 acc:",np.mean(centPTAcc),np.std(centPTAcc))
    print("Random acc:",np.mean(rdAcc),np.std(rdAcc))

    print("NB1 ",np.mean(nbLoss),np.std(nbLoss))
    print("CENT1 ",np.mean(centLoss),np.std(centLoss))
    print("PT-NB-1 ", np.mean(nbPTLoss),np.std(nbPTLoss))
    print("PT-CENT-1 ",np.mean(centPTLoss),np.std(centPTLoss))
    print("Random ",np.mean(rdLoss),np.std(rdLoss))
