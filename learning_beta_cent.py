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
        # print("lastIndividualResponsesbyQNo: ")
        # print(self.lastIndividualResponsesbyQNo)

        # Compute the group answer of the team per question
        submissions = eventLogWithAllData[(eventLogWithAllData['extra_data'] == "IndividualResponse") | (eventLogWithAllData['extra_data'] == "GroupRadioResponse") ]
        individualAnswersPerQuestion = submissions.groupby(["questionNumber","sender_subject_id"], as_index=False, sort=False).tail(1)
        # print("individualAnswersPerQuestion: ")
        # print(individualAnswersPerQuestion)

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
        self.alphas = np.ones(8, dtype = int)
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


        # print("expectedPerformance")
        # print(expectedPerformance)

        prob_class_responses = [0 for _ in range(len(self.options[idx]))]
        prob_resp_given_class = [0 for _ in range(len(self.options[idx]))]

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

            #human-agent
            # Conditional Probability
            # Do a bayes update
            denominator = 0
            numerator = [1. for _ in range(len(self.options[idx]))]
            prob_class = 0.25
            prob_rsep = 0
            
            for opt_num in range(0,len(self.options[idx])):
                numerator = prob_class

                for person_num in range(0,self.teamSize):
                    if individualResponse[person_num] == self.options[idx][opt_num]:
                        numerator *= expectedPerformance[person_num]
                    else:
                        numerator *= (1 - expectedPerformance[person_num])/3

                if machineAnswer == self.options[idx][opt_num]:
                    numerator *= expectedPerformance[4+k]
                else:
                    numerator *= (1 - expectedPerformance[4+k])/3

                prob_resp_given_class[opt_num] = numerator
            prob_class_responses = [(prob_resp_given_class[i]/sum(prob_resp_given_class)) for i in range(0,len(prob_resp_given_class))]

            # #agent alone
            # for opt_num in range(0,len(self.options[idx])):
            #     prob_class_responses[opt_num] = (1 - expectedPerformance[4+k])/3
            # for opt_num in range(0,len(self.options[idx])):
            #     if machineAnswer == self.options[idx][opt_num]:
            #         prob_class_responses[opt_num] = expectedPerformance[4+k]
            #         break

            # #random
            # for opt_num in range(4):
            #     prob_class_responses[opt_num] = 0.25

        else:
            # Conditional Probability
            # Do a bayes update
            denominator = 0
            numerator = [1. for _ in range(len(self.options[idx]))]
            prob_class = 0.25
            prob_rsep = 0

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
            # print("prob_class_responses: ")
            # print(prob_class_responses)

        #ANSIs this updating agent probabilities?
        l1 = 0
        for i in range(self.teamSize):
            l1 += expectedPerformance[self.teamSize+i]

        for i in range(self.teamSize):
            probabilities.append(1.0 * expectedPerformance[self.teamSize+i]/l1)

        #8 probability values returned
        # first set is for options (sums to 1)

        assert(sum(prob_class_responses) > 0.999 and sum(prob_class_responses) < 1.001)
        #second set is for machines
        prob_all_class_responses = prob_class_responses + [1.0 * expectedPerformance[self.getAgentForHuman(k)]/l1 for k in range(self.teamSize)]

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

    def calculatePerformanceProbability(self, questionNumber, idx, be):
        # probabilities = list()
        # probabilities = [0 for _ in range(self.teamSize + self.numAgents)]

        # for i in range(0,self.teamSize):
        #     individulResponse = self.lastIndividualResponsesbyQNo[(self.lastIndividualResponsesbyQNo["questionNumber"] == questionNumber) & (self.lastIndividualResponsesbyQNo["sender"] == self.teamMember.iloc[i])]["stringValue"].any()
        #     # print("individulResponse")
        #     # print(individulResponse)
        #     index = self.options[idx].index(individulResponse)
        #     # print("options: ")
        #     # print(self.options[idx])
        #     probabilities[index] += self.centralities[idx][i]

        # # Normalize the probabilties
        # # print("probability:")
        # # print(probabilities)
        # # totalProbability = sum(probabilities)
        # # probabilities[:] = [x / totalProbability for x in probabilities]

        # # Add expected Performance for Agents
        # for i in range(0, self.numAgents):
        #     #which agents should have a centrality of 1?
        #     if self.centralities[idx][self.getAgentForHuman(i)] == 1:
        #         probabilities[self.getAgentForHuman(i)] = self.centralities[idx][self.getAgentForHuman(i)]
        #     #which agents should have a positive centrality
        #     elif self.centralities[idx][i+self.teamSize] >= 0:
        #         probabilities[self.getAgentForHuman(i)] = self.centralities[idx][self.getAgentForHuman(i)]
        #     else:
        #         assert(False) # no negative centralities allowed

        # # print("probabilities: ")
        # # print(probabilities)

        # # l1 = 0
        # # for i in range(4,8):
        # #     l1 += probabilities[i]

        # # for i in range(4,8):
        # #     probabilities[i] /= l1 

        # # print("probability1:")
        # # print(probabilities)

        # indxQ = -1
        # anyMachineAsked = False
        # if(int(float(questionNumber)) in self.machineAskedQuestions):
        #     indxQ = self.machineAskedQuestions.index(int(float(questionNumber)))
        #     sender = self.machineAsked['sender'].iloc[indxQ]
        #     k = self.teamArray.index(sender)
        #     # print("k")
        #     # print(k)
        #     anyMachineAsked = True

        # if not anyMachineAsked:
        #     l1 = 0
        #     for i in range(0,4):
        #         l1 += probabilities[i]

        #     for i in range(0,4):
        #         probabilities[i] /= l1

        #     l1 = 0
        #     for i in range(4,8):
        #         l1 += probabilities[i]

        #     for i in range(4,8):
        #         probabilities[i] /= l1 

        # else:
        #     machineAnswer = self.machineAsked['event_content'].iloc[indxQ].split("||")[0].split(":")[2].replace('"', '').split("_")[0]
        #     ind = self.options[idx].index(machineAnswer)
        #     beta = 0

        #     for i in range(0,4):
        #         probabilities[i] *= (1-beta)
        #     probabilities[ind] += probabilities[4+k] * beta

        #     l1 = 0
        #     for i in range(0,4):
        #         l1 += probabilities[i]

        #     for i in range(0,4):
        #         probabilities[i] /= l1
        #     # print("probability2:")
        #     # print(probabilities)




        probabilities = list()
        probabilities = [0 for _ in range(self.teamSize + self.numAgents)]

        for i in range(0,self.teamSize):
            individulResponse = self.lastIndividualResponsesbyQNo[(self.lastIndividualResponsesbyQNo["questionNumber"] == questionNumber) & (self.lastIndividualResponsesbyQNo["sender"] == self.teamMember.iloc[i])]["stringValue"].any()
            # print("individulResponse")
            # print(individulResponse)
            index = self.options[idx].index(individulResponse)
            # print("options: ")
            # print(self.options[idx])
            probabilities[index] += self.centralities[idx][i]

        # Normalize the probabilties
        # print("probability:")
        # print(probabilities)
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

        # print("probabilities: ")
        # print(probabilities)

        # l1 = 0
        # for i in range(4,8):
        #     l1 += probabilities[i]

        # for i in range(4,8):
        #     probabilities[i] /= l1 




        # print("probability1:")
        # print(probabilities)

        indxQ = -1
        anyMachineAsked = False
        if(int(float(questionNumber)) in self.machineAskedQuestions):
            indxQ = self.machineAskedQuestions.index(int(float(questionNumber)))
            sender = self.machineAsked['sender'].iloc[indxQ]
            k = self.teamArray.index(sender)
            # print("k")
            # print(k)
            anyMachineAsked = True

        if anyMachineAsked:
            machineAnswer = self.machineAsked['event_content'].iloc[indxQ].split("||")[0].split(":")[2].replace('"', '').split("_")[0]
            ind = self.options[idx].index(machineAnswer)

            #cent-agent
            for i in range(0,4):
               probabilities[i] *= (1-be)
            probabilities[ind] += probabilities[4+k] * be

            # probabilities[ind] += probabilities[4+k]

            l1 = 0
            for i in range(0,4):
                l1 += probabilities[i]

            for i in range(0,4):
                probabilities[i] /= l1

            # print("probability2:")
            # print(probabilities)

            # #agent
            # for i in range(0,4):
            #     probabilities[i] = 1.0 * (1-probabilities[4+k]) / 3
            # probabilities[ind] = probabilities[4+k]

            # #random
            # for i in range(0,4):
            #     probabilities[i] = 0.25

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

    def calculateRewards(self,be):
        rewardsNB1 = list()
        probabilitiesNB1 = list()
        group_accuracies = list()
        group_accuracy_per_question = list() # for each question
        expectedP = list()
        h1 = list()
        h2 = list()
        h3 = list()
        h4 = list()
        
        # Compute Reward for NB1
        for i in range(0,self.numQuestions):
            expectedPerformance, all_probabilities, human_accuracy, group_accuracy, machine_accuracy = self.naiveProbability(self.questionNumbers[i],i)
            group_accuracy_per_question.append(group_accuracy)
            expectedP.append(expectedPerformance)
            h1.append(human_accuracy[0])
            h2.append(human_accuracy[1])
            h3.append(human_accuracy[2])
            h4.append(human_accuracy[3])
            probabilitiesNB1.append(all_probabilities)
            rewardsNB1.append(self.calculateExpectedReward(all_probabilities))
            # print("expectedPerformance: ")
            # print(expectedPerformance)
            # print("all_probabilities: ")
            # print(all_probabilities)
            # print("human_accuracy: ")
            # print(human_accuracy)
            # print("machine_accuracy: ")
            # print(machine_accuracy)
        # print(rewardsNB1)

        #Compute Reward for CENT1 model
        rewardsCENT1 = list()
        probabilitiesCENT1 = list()

        for i in range(0,self.numCentralityReports):
            self.updateCentrality(self.influenceMatrixIndex)

        # print("centrality: ")
        # print(self.centralities)

        for i in range(0,self.numQuestions):
            probabilities = self.calculatePerformanceProbability(self.questionNumbers[i],i,be)
            probabilitiesCENT1.append(probabilities)
            rewardsCENT1.append(self.calculateExpectedReward(probabilities))

        best_human_accuracy = max([sum(h1)/len(h1), sum(h2)/len(h2), sum(h3)/len(h3), sum(h4)/len(h4)])
        return expectedP, rewardsNB1,rewardsCENT1, probabilitiesNB1,probabilitiesCENT1, group_accuracy_per_question, best_human_accuracy

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
            # elif int(float(self.questionNumbers[i])) in self.machineAskedQuestions:
            #     self.actionTaken.append(self.teamSize + np.where(self.machineUsed[i] == True)[0][0])
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
    def computeAgentLoss(self,params,probabilities,chosen_action,lossType,loss_function,modelName):
        current_models = ["nb","nb-pt","cent","cent-pt"]
        if (modelName not in current_models):
            assert(False)

        prospects= []

        for probability in probabilities:
            prospectSuccess = self.c +self.e, probability
            prospectFailure = self.z +self.e, 1-probability
            prospects.append((prospectSuccess,prospectFailure))
        # print("agent: ")
        # print(prospects)
        # print(params)
        cpt_vals = evaluateProspectVals(params,prospects)
        # print(cpt_vals)
        # print(chosen_action)

        soft_prob = self.softmax(cpt_vals)
        # print("agent_prob:")
        # print(soft_prob)
        predicted = np.argmax(soft_prob)
        ground = chosen_action

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
                for k in range(4):
                    arg = soft_prob[k]
                    loss += -arg * math.log(arg)
            else:
                # print("predicted:")
                # print(i)
                # print("chosen_action:")
                # print(j)
                index1 = []
                index2 = []
                for k in range(4):
                    if soft_prob[k] >= soft_prob[j]:
                        index1.append(k)
                    else:
                        index2.append(k)
                m = 0
                for ind in index1:
                    m += soft_prob[ind]
                
                variation_q =  np.zeros(4)
                for ind in index1:
                    variation_q[ind] = 1.0 * m / len(index1)

                for ind in index2:
                    variation_q[ind] = soft_prob[ind]

                for k in range(4):
                    loss += -soft_prob[k] * math.log(variation_q[k])
        else:
            i = np.argmax(soft_prob)
            j = chosen_action
            if i == j:
                for k in range(4):
                    arg = soft_prob[k]
                    loss += -arg * math.log(arg)
            else:

                index1 = []
                index2 = []
                for k in range(4):
                    if soft_prob[k] >= soft_prob[j]:
                        index1.append(k)
                    else:
                        index2.append(k)
                
                variation_q =  np.zeros(4)
                for ind in index1:
                    variation_q[ind] = 1.0  / len(index1)

                for k in range(4):
                    loss += -variation_q[k] * math.log(soft_prob[k])


        return loss, acc, predicted, ground


    #action in 0,...,numOptions-1
    def computeHumanLoss(self,params,probabilities,chosen_action,lossType,loss_function,modelName):
        current_models = ["nb","nb-pt","cent","cent-pt"]
        if (modelName not in current_models):
            assert(False)
        prospects= []
        for probability in probabilities:
            prospectSuccess = self.c, probability
            prospectFailure = self.z, 1-probability
            prospects.append((prospectSuccess,prospectFailure))
        # print("human: ")
        # print(prospects)

        cpt_vals = evaluateProspectVals(params,prospects)
        # print(cpt_vals)
        # print(chosen_action)
        soft_prob = self.softmax(cpt_vals)
        # print("human_prob:")
        # print(soft_prob)
        predicted = np.argmax(soft_prob)
        ground = chosen_action
        acc = 0
        if predicted == ground:
            acc = 1

        loss = 0
        if loss_function == "binary":
            arg = soft_prob[chosen_action]
            # print(arg)
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
                for k in range(4):
                    arg = soft_prob[k]
                    loss += -arg * math.log(arg)
            else:
                # print("predicted:")
                # print(i)
                # print("chosen_action:")
                # print(j)
                index1 = []
                index2 = []
                for k in range(4):
                    if soft_prob[k] >= soft_prob[j]:
                        index1.append(k)
                    else:
                        index2.append(k)
                m = 0
                for ind in index1:
                    m += soft_prob[ind]
                
                variation_q =  np.zeros(4)
                for ind in index1:
                    variation_q[ind] = 1.0 * m / len(index1)

                for ind in index2:
                    variation_q[ind] = soft_prob[ind]

                for k in range(4):
                    loss += -soft_prob[k] * math.log(variation_q[k])
        else:
            i = np.argmax(soft_prob)
            j = chosen_action
            if i == j:
                for k in range(4):
                    arg = soft_prob[k]
                    loss += -arg * math.log(arg)
            else:

                index1 = []
                index2 = []
                for k in range(4):
                    if soft_prob[k] >= soft_prob[j]:
                        index1.append(k)
                    else:
                        index2.append(k)
                
                variation_q =  np.zeros(4)
                for ind in index1:
                    variation_q[ind] = 1.0  / len(index1)

                for k in range(4):
                    loss += -variation_q[k] * math.log(soft_prob[k])

        return loss, acc, predicted, ground


    def computeCPTLoss(self,params,probabilities,lossType,loss_function,modelName):
        total_loss = 0
        per_question_agent_loss = [None for _ in range(self.numQuestions)]
        per_question_option_loss = [None for _ in range(self.numQuestions)]
        per_question_ground = [None for _ in range(self.numQuestions)]
        per_question_predicted = [None for _ in range(self.numQuestions)]

        per_question_agent_acc = [None for _ in range(self.numQuestions)]
        per_question_option_acc = [None for _ in range(self.numQuestions)]
        # print(params)

        length = len(probabilities)


        start = 0
        if length==self.testSetSize:
            start = self.trainingSetSize
        
        for question_num in range(length):
            agent_loss  = False
            for is_used in self.machineUsed[start+question_num]:
                
                if (is_used == True):
                    #compute agent loss
                    # print("question_num: ")
                    # print(start+question_num)
                    agent_loss = True
                    break
            # Here - How to handle consensus not reached case

            if self.actionTaken[start+question_num]==-1:
                continue
            if (agent_loss):
            #     assert(self.actionTaken[start+question_num] in range(self.teamSize,self.teamSize+self.numAgents))
            #     loss_agent, acc_agent, predicted, ground = self.computeAgentLoss(params[1],probabilities[question_num][self.teamSize:],(self.actionTaken[start+question_num]-self.teamSize),lossType,loss_function,modelName)
            #     per_question_agent_loss[start+question_num] = loss_agent
            #     per_question_agent_acc[start+question_num] = acc_agent
            #     per_question_ground[start+question_num] = ground + 4
            #     per_question_predicted[start+question_num] = predicted + 4
            # else:
                assert(self.actionTaken[start+question_num] < len(self.options[start+question_num]))
                loss_option, acc_option, predicted, ground = self.computeHumanLoss(params[0],probabilities[question_num][0:self.teamSize],self.actionTaken[start+question_num],lossType,loss_function,modelName)
                per_question_option_loss[start+question_num] = loss_option
                per_question_option_acc[start+question_num] = acc_option
                per_question_ground[start+question_num] = ground
                per_question_predicted[start+question_num] = predicted
        # print("agent: ")
        # print(per_question_agent_loss)
        # print(per_question_option_loss)

        return per_question_option_loss,per_question_agent_loss,per_question_option_acc,per_question_agent_acc,per_question_ground,per_question_predicted


    def computeAverageLossPerTeam(self,params, probabilities, lossType, loss_function, modelName):
        # print(params)
  
        # agent_prob = np.array([0.25,0.25,0.25,0.25])
        # human_prob = np.array([0.25,0.25,0.25,0.25])
        # human_prob = np.asarray(params)[:4]

        if modelName == "random":

            per_question_agent_loss = [None for _ in range(self.numQuestions)]
            per_question_option_loss = [None for _ in range(self.numQuestions)]
            per_question_agent_acc = [None for _ in range(self.numQuestions)]
            per_question_option_acc = [None for _ in range(self.numQuestions)]

            length = len(probabilities)
            start = 0
            if length==self.testSetSize:
                start = self.trainingSetSize
        

            for question_num in range(length):
                agent_loss  = False
                for is_used in self.machineUsed[start+question_num]:
                    if (is_used == True):
                        #compute agent loss
                        agent_loss = True
                        break
                #Here - How to handle consensus not reached case
                if self.actionTaken[start+question_num]==-1:
                    continue
                if (agent_loss):
                    assert(self.actionTaken[start+question_num] in range(self.teamSize,self.teamSize+self.numAgents))
                    # r = random.randint(0,3)
                    # prob = agent_prob[r]
                    prob = 0.25
                    per_question_agent_loss[start+question_num] = -1.0*math.log(prob)
                    per_question_agent_acc[start+question_num] = 0.25
                else:
                    assert(self.actionTaken[start+question_num] < len(self.options[start+question_num]))
                    # r = random.randint(0,3)
                    # prob = human_prob[r]
                    prob = 0.25
                    per_question_option_loss[start+question_num] = -1.0*math.log(prob)
                    per_question_option_acc[start+question_num] = 0.25


        else:

            (per_question_option_loss, per_question_agent_loss,per_question_option_acc,per_question_agent_acc,ground,predicted) = self.computeCPTLoss(params,probabilities,lossType,loss_function,modelName)



        agent_loss = 0
        option_loss = 0
        agent_count = 0
        option_count = 0

        agent_acc = 0
        option_acc = 0
        agent_count_acc = 0
        option_count_acc = 0


        for (optionLoss,agentLoss) in zip(per_question_option_loss,per_question_agent_loss):
            if (optionLoss != None):
                option_loss += optionLoss
                option_count += 1
            if (agentLoss != None):
                agent_loss += agentLoss
                agent_count += 1
            #If consensus is not reached, it is ignored
            #assert((agentLoss == None) ^ (optionLoss == None))
            assert((agentLoss==None) |(optionLoss== None))

        if option_count!=0:
            option_loss /= option_count
        if agent_count!=0:
            agent_loss /= agent_count

        for (optionAcc,agentAcc) in zip(per_question_option_acc,per_question_agent_acc):
            if (optionAcc != None):
                option_acc += optionAcc
                option_count_acc += 1
            if (agentAcc != None):
                agent_acc += agentAcc
                agent_count_acc += 1
            #If consensus is not reached, it is ignored
            #assert((agentLoss == None) ^ (optionLoss == None))
            assert((agentAcc==None) |(optionAcc== None))

        if option_count_acc!=0:
            option_acc /= option_count_acc
        if agent_count_acc!=0:
            agent_acc /= agent_count_acc

        ground1 = []
        predicted1 = []
        for ele in ground:
            if ele != None:
                ground1.append(ele)

        for ele in predicted:
            if ele != None:
                predicted1.append(ele)


        return agent_loss + option_loss, option_loss, agent_loss, option_acc, agent_acc, ground1,predicted1


    def chooseCPTParameters(self, probabilities,lossType,loss_function,modelName):
        # hAlpha, hGamma,hLambda =  (None,None,None)
        best_parameters = []
        option_alpha, option_beta, option_lambda, option_gammaGain, option_gammaLoss =  (None,None,None,None,None)
        agent_alpha, agent_beta, agent_lambda, agent_gammaGain, agent_gammaLoss =  (None,None,None,None,None)
        Alpha, Beta, Lambda, GammaGain, GammaLoss =  (None,None,None,None,None)
        option_loss = np.float("Inf")
        agent_loss = np.float("Inf")
        Loss = np.float("Inf")

        for alpha in np.arange(0,1.1,0.1):
            for lamda in np.arange(1,11,1):
                for gammaGain in np.arange(0,1.1,0.1):
                    for gammaLoss in np.arange(0,1.1,0.1):
            # for gamma in np.arange(0,1.05,0.05):
                # for lamb in np.arange(0.1,5,0.1):
                        parameters = []
                        parameters.append((alpha,alpha,lamda,gammaGain,gammaLoss))
                        parameters.append((alpha,alpha,lamda,gammaGain,gammaLoss))

                        loss_cpt,option_loss_cpt, agent_loss_cpt, option_acc, agent_acc, ground,predicted = self.computeAverageLossPerTeam(parameters,probabilities,lossType,loss_function,modelName)

                        if (option_loss_cpt<option_loss):
                            option_loss = option_loss_cpt
                            option_alpha = alpha
                            option_beta = alpha
                            option_lambda = lamda
                            option_gammaGain = gammaGain
                            option_gammaLoss = gammaLoss

                        if (agent_loss_cpt<agent_loss):
                            agent_loss = agent_loss_cpt
                            agent_alpha = alpha
                            agent_beta = alpha
                            agent_lambda = lamda
                            agent_gammaGain = gammaGain
                            agent_gammaLoss = gammaLoss

                        if (loss_cpt<Loss):
                            Loss = loss_cpt
                            Alpha = alpha
                            Beta = alpha
                            Lambda = lamda
                            GammaGain = gammaGain
                            GammaLoss = gammaLoss


        assert(option_alpha != None)
        assert(option_beta != None)
        assert(option_lambda != None)
        assert(option_gammaGain != None)
        assert(option_gammaLoss != None)
        assert(agent_alpha != None)
        assert(agent_beta != None)
        assert(agent_lambda != None)
        assert(agent_gammaGain != None)
        assert(agent_gammaLoss != None)
        assert(Alpha != None)
        assert(Beta != None)
        assert(Lambda != None)
        assert(GammaGain != None)
        assert(GammaLoss != None)
        # assert(hLambda != None)

        best_parameters.append((option_alpha,option_beta,option_lambda,option_gammaGain,option_gammaLoss))
        best_parameters.append((agent_alpha,agent_beta,agent_lambda,agent_gammaGain,agent_gammaLoss))

        # best_parameters.append((Alpha,Gamma,1))
        # best_parameters.append((Alpha,Gamma,1))

        # return (hAlpha, hGamma,hLambda)
        return best_parameters

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
    nbAlpha = list()
    nbGamma = list()
    centAlpha = list()
    centGamma = list()
    
    rdOptionLoss = list()
    rdAgentLoss = list()

    nbOptionLoss = list()
    nbAgentLoss = list()
    nbOptionAcc = list()
    nbAgentAcc = list()

    OptionPre = list()
    AgentPre = list()
    OptionRe = list()
    AgentRe = list()
    OptionF1 = list()
    AgentF1 = list()

    centOptionLoss = list()
    centAgentLoss = list()
    centOptionAcc = list()
    centAgentAcc = list()

    nbPTOptionLoss = list()
    nbPTAgentLoss = list()
    nbPTOptionAcc = list()
    nbPTAgentAcc = list()

    centPTOptionLoss = list()
    centPTAgentLoss = list()
    centPTOptionAcc = list()
    centPTAgentAcc = list()

    total_group_accuracy = list()
    teamAccuracies = list()
    allBestHumanAccuracy = list()
    group_accuracy_per_question = list()
    group_accuracy_over_time = np.zeros(NUM_QUESTIONS)

    #lossType = "logit"
    lossType = "softmax"

    #loss_function = "binary"
    #loss_function = "variational_H_qp"
    loss_function = "variational_H_pq"
    option_predicted = []
    option_ground = []
    agent_predicted = []
    agent_ground = []

    ind_team = [1, 2, 3, 4, 8, 10, 11, 15, 16, 19, 20, 22, 23, 24, 25, 27, 28, 29, 30, 31, 34, 35, 46, 47, 48, 51, 52, 53, 55, 56]

    #shuffe_ind_team = np.random.permutation(ind_team);
    shuffe_ind_team = np.asarray([28,55,31,29,48,20,3,47,56,16,25,34,8,24,52,15,22,11,10,4,35,53,30,23,46,27,2,51,1,19])
    print("shuffled index: ")
    print(shuffe_ind_team)
    bee = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    all_loss = []


    for be in bee:
        for ind in range(20):    

            i = shuffe_ind_team[ind]
            print("i: ")
            print(i)
            print("Values of team", team.iloc[i]['id'])

            #Create model
            model = HumanDecisionModels(team.iloc[i]['id'], directory)

            expectedPerformance, rewardsNB1, rewardsCENT1,probabilitiesNB1,probabilitiesCENT1, group_accuracy_per_question, best_human_accuracy = model.calculateRewards(be)

            parameters = []
            parameters.append(RATIONAL_PARAMS)
            parameters.append(RATIONAL_PARAMS)

            loss, optionLoss, agentLoss, optionAcc, agentAcc, ground,predicted = model.computeAverageLossPerTeam(parameters,probabilitiesCENT1,lossType,loss_function,"cent")
            if optionLoss != 0:
                centOptionLoss.append(optionLoss)
            centAgentLoss.append(agentLoss)
            centLoss.append(loss)
            centOptionAcc.append(optionAcc)
            centAgentAcc.append(agentAcc)

        all_loss.append(np.mean(centOptionLoss))
    min_index = np.argmin(np.asarray(all_loss))

    be = bee[min_index]
    print("the best beta:")
    print(be)

    centOptionLoss = []

    for ind in range(20,30):    

        i = shuffe_ind_team[ind]
        print("i: ")
        print(i)
        print("Values of team", team.iloc[i]['id'])

        #Create model
        model = HumanDecisionModels(team.iloc[i]['id'], directory)

        expectedPerformance, rewardsNB1, rewardsCENT1,probabilitiesNB1,probabilitiesCENT1, group_accuracy_per_question, best_human_accuracy = model.calculateRewards(be)

        parameters = []
        parameters.append(RATIONAL_PARAMS)
        parameters.append(RATIONAL_PARAMS)

        loss, optionLoss, agentLoss, optionAcc, agentAcc, ground,predicted = model.computeAverageLossPerTeam(parameters,probabilitiesCENT1,lossType,loss_function,"cent")
        if optionLoss != 0:
            centOptionLoss.append(optionLoss)

    print("CENT1 ",np.mean(centOptionLoss),np.std(centOptionLoss))


