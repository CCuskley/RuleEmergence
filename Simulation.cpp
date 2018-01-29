
//  Created by Christine Cuskley on 17/09/2014.



#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <list>
#include <cmath>
#include <random>
#include <map>

int popSize;//The starting size of the population
int runningPopSize;//the running population size (this changes over time for growth)
int k=1500;//token threshold for proficiency
const float r = 0.001; //the rate of replacement for turnover
const float g = 0.001;//the rate of growth
//replacement and growth:
//at each INTERACTION
//for replacement: at each interaction, there is a r chance a random learner will become a new learner
//for growth, at each interaction, there is a g chanec that a new learner will be ADDED

bool growth;//does this simulation include growth?
bool replacement;//does this simulation include replacement (turnover)
//These values are set at runtime



int popSteps = 10000;//number of timesteps to run the simulation for
const int fWindow=100;//value for timesteps elapsed before an agent forgets a lemma/inflection pairing
int top = 0;
int allTokens=0;
//list of 500 tokens of 28 lemma types in zipfian distribtion, generated with genVocab.py
//each lemma is identified by an index between 0-27
int vocList[500] ={1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 2, 23, 1, 1, 24, 1, 9, 1, 1, 23, 1, 1, 1, 9, 9, 1, 1, 26, 1, 0, 13, 1, 1, 23, 0, 1, 23, 9, 1, 1, 9, 23, 1, 1, 1, 7, 23, 1, 23, 0, 0, 23, 1, 1, 1, 0, 1, 1, 1, 23, 1, 1, 23, 0, 1, 23, 1, 0, 1, 1, 19, 1, 9, 1, 1, 1, 23, 15, 1, 1, 0, 1, 1, 0, 1, 1, 1, 9, 1, 1, 9, 1, 1, 0, 1, 1, 9, 15, 1, 1, 1, 1, 13, 1, 1, 0, 1, 9, 1, 0, 23, 18, 1, 1, 20, 23, 0, 23, 0, 1, 0, 1, 22, 1, 1, 1, 1, 1, 1, 1, 0, 1, 23, 1, 1, 1, 1, 23, 1, 9, 0, 1, 1, 1, 1, 1, 0, 1, 1, 13, 1, 2, 0, 1, 1, 1, 1, 1, 11, 1, 0, 1, 1, 0, 0, 1, 0, 0, 9, 0, 1, 4, 0, 1, 15, 0, 1, 1, 1, 1, 9, 11, 1, 0, 0, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 23, 0, 1, 7, 4, 1, 1, 1, 0, 1, 0, 1, 23, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 15, 0, 1, 1, 18, 1, 18, 1, 1, 1, 15, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 1, 1, 1, 15, 1, 1, 1, 1, 9, 9, 15, 1, 1, 1, 1, 1, 0, 9, 5, 1, 1, 1, 1, 1, 18, 8, 1, 0, 1, 23, 1, 0, 1, 23, 1, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1, 1, 9, 23, 1, 1, 1, 17, 11, 10, 15, 1, 1, 23, 15, 1, 0, 23, 7, 1, 1, 1, 1, 1, 1, 13, 25, 0, 1, 1, 1, 1, 1, 1, 1, 1, 6, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 11, 21, 1, 1, 9, 1, 1, 1, 1, 0, 1, 1, 3, 1, 1, 1, 1, 1, 0, 23, 19, 14, 26, 15, 1, 1, 1, 1, 23, 1, 1, 1, 1, 1, 1, 15, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 23, 1, 23, 1, 0, 23, 1, 1, 1, 11, 23, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 15, 1, 1, 23, 1, 15, 1, 1, 0, 1, 1, 0, 1, 0, 1, 23, 1, 12, 1, 1, 1, 23, 1, 1, 0, 1, 1, 9, 1, 23, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 13, 1, 1, 9, 0, 1, 1, 1, 9, 0, 1, 0, 1, 27, 1, 15, 1, 1, 1, 1, 23, 0, 1, 1, 0, 1, 1, 1, 1, 19, 13, 0, 6, 1, 1, 1, 1};

//simulations in Cuskley, Kirby, & Loreto (2018) were averaged over 100 runs
int totalRuns;

//Keeps track of the frequency of each lemma throughout the simluation
int globCounts[28];

//Keeps track of the frequency of each inflection throughout the simluation
int globInfls[12];

//functions for running the simulation, defined later
//each timeStep is popSize interactions
//each run is popSteps long (10000)
void interaction (int s, int h, int l);
void timeStep(int tNow);
void singleRun(int run);

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0,1);


//class which defines an inflection as paired with a lemma
//stores number of interactions, successes, weight (successes/interactions)
//and lastInter, which is the timestep when the pairing was last encountered, later compared against fWindow
//weight and lastInter are initialised as negative to indicate the pairing is non-existent
struct Inflection {
    //char infl;
    int inters = 0;
    int successes = 0;
    float weight = -1;
    int lastInter = -1;
    
    void emptyInflection() {
        inters=0;
        successes=0;
        weight=-1;
        lastInter=-1;
    }
};

//Lemma class
class Lemma {
public:
    int lemma; //the int which defines the lemma index
    int tokens =0;//number of times agent has encountered this lemma
    bool seenLemma = false;//whether the agent has yet encountered this lemma
    Inflection inflections [12];//list of potential inflections for the lemma
    
    void resetLemma() {
        tokens=0;
        seenLemma=false;
        for (int i=0; i<12; i++) {
            inflections[i].emptyInflection();
        }
    }
    void addInflection(int infl, int outcome, int tstep) {
        seenLemma = true;
        tokens=1;
        inflections[infl].inters=1;
        inflections[infl].successes=outcome;
        inflections[infl].weight=float(outcome)/float(inflections[infl].inters);
        inflections[infl].lastInter=tstep;
    }
    
    void updateInflection(int infl, int outcome, int tstep) {
        ++tokens;
        inflections[infl].inters+=1;
        inflections[infl].successes+=outcome;
        inflections[infl].weight=float(inflections[infl].successes)/float(inflections[infl].inters);
        inflections[infl].lastInter=tstep;
    }
    
    //test if agent already has a specific inflection for this lemma
    bool hasInflection(int infl) {
        if (inflections[infl].inters>0) {
            return true;
        } else {
            return false;
        }
    }
    
    //return the highest weighted inflection for this lemma
    int getBest() {
        int bestInfl = 0;
        float bestWeight = -0.5;
        for (int i=0; i<12;i++) {
            if (inflections[i].weight>bestWeight) {
                bestWeight=inflections[i].weight;
                bestInfl=i;
            }
        }
        if (bestWeight == -0.5) {
            return -1;
        } else {
            return bestInfl;
        }
    }
    
    //test if this agent has any active inflections for this lemma
    bool hasAnyInflection() {
        bool hasints=false;
        for (int i=0; i<12;i++) {
            if (inflections[i].inters > 0 ) {
                hasints=true;
                break;
            }
        }
        return hasints;
    }
    
    //purge inflections from this lemma based on fWindow lapse
    void purge(int tstep) {
        for (int i=0; i<12;i++) {
            if (tstep - inflections[i].lastInter > fWindow) {
                Inflection empty;
                inflections[i] = empty;
                
            }
        }
    }
    
};

//Agent class
class Agent {
public:
    
    int tokens;
    Lemma voc[28];//initialise empty vocabulary
    int tokenThresh;
    int dWindow;
    bool typeGeneralise;
    bool isActive;
    Agent() {
        isActive=false;
        tokens=0;
        typeGeneralise=false;//initialise agents as token generalisers
        tokenThresh=k;
        dWindow=100;
    }
    
    void resetAgent() {
        isActive=true;
        tokens=0;
        typeGeneralise=false;
        for (int i=0;i<28;i++) {
            voc[i].resetLemma();
        }
    }
    
    //test if agent has any inflections for a particular lemma
    bool hasInflections(int lemint) {
        return voc[lemint].hasAnyInflection();
    }
    
    //update the entry for a particular Lemma
    void updateLemma(int lemint, int infl, int outcome, int tstep) {
        tokens+=1;
        if (voc[lemint].hasInflection(infl)) {
            voc[lemint].updateInflection(infl, outcome, tstep);
        } else {
            voc[lemint].addInflection(infl,outcome,tstep);
        }
        voc[lemint].purge(tstep);
        if (tokens>tokenThresh) {
            typeGeneralise = true;
        } else {
            typeGeneralise = false;
        }
    }
    
    //take inflection in as a hearer
    int hear(int lemint,int infl, int tstep) {
        //if there are any inflections present for this lemma/agent
        if (hasInflections(lemint)) {
            //if this agent has this inflection for this lemma - no matter the weight - return success.
            //or return success with likelihood proporitional to weight?
            if (voc[lemint].hasInflection(infl)) {
                updateLemma(lemint, infl, 1, tstep);
                return 1;
            } else {
                updateLemma(lemint,infl, 0, tstep);
                return 0;
            }
        } else {
            int guess = genInfl(lemint);
            if (guess==infl) {
                updateLemma(lemint, infl, 1, tstep);
                return 1;
            } else {
                updateLemma(lemint, infl, 0, tstep);
                return 0;
            }
        }
        
    }
    
    int getBest(int lemint) {
        return voc[lemint].getBest();
    }
    
    //token generalise
    int getTokenBest() {
        int maxTokens [12] = {0,0,0,0,0,0,0,0,0,0};
        for (int l=0; l<28;l++) {
            for (int i=0; i<10;i++) {
                maxTokens[i] += voc[l].inflections[i].successes;
            }
        }
        int maxIndex = -1;
        int maxVal = 0;
        for  (int j=0; j<10;j++) {
            if (maxTokens[j]>maxVal) {
                maxVal=maxTokens[j];
                maxIndex=j;
            }
        }
        return maxIndex;
    }
    
    //type generalise
    int getTypeBest() {
        int maxTypes [12] = {0,0,0,0,0,0,0,0,0,0};
        for (int l=0; l<28; l++) {
            int bestInfl=voc[l].getBest();
            maxTypes[bestInfl]+=1;
        }
        int maxIndex = -1;
        int maxVal = 0;
        for  (int j=0; j<10;j++) {
            if (maxTypes[j]>maxVal) {
                maxVal=maxTypes[j];
                maxIndex=j;
            }
        }
        return maxIndex;
    }
    
    //if a lemma has no inflections, generate an inflection based on generalisation processes
    int genInfl(int lemint) {        int inflUtt = -1;
        if (typeGeneralise) {
            inflUtt=getTypeBest();
            if (inflUtt == -1) {
                inflUtt=getTokenBest();
            } else {
            }
        } else {
            inflUtt=getTokenBest();
            if (inflUtt== -1) {
                inflUtt=getTypeBest();
            }
        }
        if (inflUtt == -1) {
            inflUtt=int(round(dis(gen)*12));
            
        }
        return inflUtt;
    }
    
};

std::vector <Agent> pop(3000);
//Agent pop[2000];

//function to run main simulation; runs from bash script doruns.sh
int main(int argc, char **argv)
{
    srand( static_cast<unsigned int>(time(NULL)));
    totalRuns = atof(argv[1]);
    replacement = atof(argv[2]);
    growth= atof(argv[3]);
    popSize = atof(argv[4]);
    runningPopSize = atof(argv[4]);
    //set initial population as active
    
    
    for (int i =0;i<popSize;i++) {
        pop[i].isActive=1;
    }

    std::cout<<"Replacement is set to ";
    std::cout<<replacement;
    std::cout<<"; Growth is set to ";
    std::cout<<growth;
    std::cout<<std::endl;
    std::cout<<"Commencing ";
    std::cout<<totalRuns;
    std::cout<<" runs with starting population size of ";
    std::cout<<popSize;
    std::cout<<" agents.";
    std::cout<<std::endl;
    
    for (int i=0; i<totalRuns; i++) {

        allTokens=0;
        //int globInfls[12];
        //int globCounts[12]
        for (int i=0;i<12;i++) {
            globInfls[i]=0;
        }
        for (int i=0;i<28;i++) {
            globCounts[i]=0;
        }
        std::vector <Agent> pop(2000);
        runningPopSize=popSize;
        std::cout<<"Starting run ";
        std::cout<<i;
        std::cout<<" with population size of ";
        std::cout<<popSize;
        std::cout<<" and running population size of ";
        std::cout<<runningPopSize;
        std::cout<<std::endl;

        singleRun(i);
        std::cout<<"Run number ";
        std::cout<<i;
        std::cout<<" is complete. Ended with ";
        std::cout<<runningPopSize;
        std::cout<<" total agents.";
        std::cout<<std::endl;
    }
    return 0;
}

//replace an agent in turnover
void replaceAgent() {
    int chosenOne = int(round(dis(gen)*runningPopSize));
    float roll = dis(gen);
    if (roll<=r) {
        //Agent nagent;
        //pop[chosenOne] = nagent;
        //pop[chosenOne].isActive=true;
        pop[chosenOne].resetAgent();
    }
}

//add an agent in growth
void addAgent() {
    
    float roll = dis(gen);
    if (roll<=g) {
        //std::cout<<"Adding agent...there are now ";
        //std::cout<<runningPopSize;
        //std::cout<<" agents.";
        //std::cout<<std::endl;
        runningPopSize+=1;
        pop[runningPopSize-1].isActive=1;
        //std::cout<<"Agent addition successful!";
        //std::cout<<std::endl;
    }
    
}

//interaction function
void interaction(int s,int h, int lem, int tNow) {
    int utterance;
    int result;
    
    //std::cout<<"Testing speaker for inflections for this lemma...";
   // std::cout<<std::endl;
    if (pop[s].hasInflections(lem)) {//if lemma has inflections
        //std::cout<<"Speaker has inflections, getting best...";
        //std::cout<<std::endl;
        utterance=pop[s].getBest(lem);
        //std::cout<<"Best acquired, evaluating interaction...";
        //std::cout<<std::endl;
        result=pop[h].hear(lem,utterance,tNow);
        //std::cout<<"Evaluation successful...";
        //std::cout<<std::endl;
    } else {
        
        //std::cout<<"No inflections, generating...";
        //std::cout<<std::endl;
        utterance=pop[s].genInfl(lem);
        //std::cout<<"Utterance generated, evaluating interaction...";
        //std::cout<<std::endl;
        result=pop[h].hear(lem,utterance,tNow);
    }
    //std::cout<<"Updating lemma for speaker...";
    //std::cout<<std::endl;
    pop[s].updateLemma(lem,utterance,result,tNow);
    //std::cout<<"Speaker updated. Updating counts...";
    //std::cout<<std::endl;
    globInfls[utterance]+=1;
    
    
}


//counts inflections for typeGenerlisers, token generalisers, and the whole population
//only the whole population count turns out to be relevant
int inflsInVoc(int learnType) {
    int inflProbs[] = {0,0,0,0,0,0,0,0,0,0,0,0};
    int totinfls=0;
    if (learnType==1){
        for (int l = 0; l<28;l++) {
            for (int a=0;a<runningPopSize;a++) {
                if (pop[a].hasInflections(l) && pop[a].typeGeneralise) {
                    int best;
                    best=pop[a].getBest(l);
                    inflProbs[best]+=1;
                }
            }
        }
    } else if (learnType==2) {
        for (int l = 0; l<28;l++) {
            for (int a=0;a<runningPopSize;a++) {
                if (pop[a].hasInflections(l) && !pop[a].typeGeneralise) {
                    int best;
                    best=pop[a].getBest(l);
                    inflProbs[best]+=1;
                }
            }
        }
    } else {
        for (int l = 0; l<28;l++) {
            for (int a=0;a<runningPopSize;a++) {
                if (pop[a].hasInflections(l)) {
                    int best;
                    best=pop[a].getBest(l);
                    inflProbs[best]+=1;
                }
            }
        }
    }
    
    //now i have an array, index is infl
    //count at each location is number of agents with that
    //if count>0,totinfls++
    for (int i = 0; i<12;i++) {
        if (inflProbs[i]>0) {
            totinfls+=1;
        }
    }
    return totinfls;
}

//calculates the entropy from a list of probablities/frequencies
float getEntropy(float problist[12]) {
    float actEntropy = 0;
    for (int i = 0; i<12; i++) {
        if (problist[i] > 0) {
            actEntropy += (problist[i]*log2f(problist[i]));
        }
    }
    actEntropy = -actEntropy;
    return actEntropy;
}

//calculates entropy of inflection across the vocabulary, H_v
float vocabEntropy(int learnType) {
    float inflProbs[] = {0,0,0,0,0,0,0,0,0,0,0,0};
    int pdenom=0;
    //H_v
    //how predictable is the inflection of any given lemma?
    //for each lemma
    if (learnType==1) {
        for (int l = 0; l<28;l++) {
            for (int a=0;a<runningPopSize;a++) {
                if (pop[a].voc[l].hasAnyInflection() && pop[a].typeGeneralise) {
                    pdenom+=1;
                    int best;
                    best=pop[a].getBest(l);
                    inflProbs[best]+=1;
                }
            }
        }
    } else if (learnType==2) {
        for (int l = 0; l<28;l++) {
            for (int a=0;a<runningPopSize;a++) {
                if (pop[a].voc[l].hasAnyInflection() && !pop[a].typeGeneralise) {
                    pdenom+=1;
                    int best;
                    best=pop[a].getBest(l);
                    inflProbs[best]+=1;
                }
            }
        }
    } else {
        for (int l = 0; l<28;l++) {
            for (int a=0;a<runningPopSize;a++) {
                if (pop[a].voc[l].hasAnyInflection()) {
                    pdenom+=1;
                    int best;
                    best=pop[a].getBest(l);
                    inflProbs[best]+=1;
                }
            }
        }
    }
    
    for (int i=0; i<12;i++) {
        inflProbs[i] = inflProbs[i]/float(pdenom);
    }
    return getEntropy(inflProbs);
}

//calculates entropy of the inflection for a specific lemma, H_l
float meaningEntropy(int lem, int learnType) {
    //what is the probability of each inflection given this lemma?
    int inflcts [] = {0,0,0,0,0,0,0,0,0,0,0,0};
    int lemct = 0;
    float inflProbs [] = {0,0,0,0,0,0,0,0,0,0,0,0};
    if (learnType==1) {
        for (int a=0;a<runningPopSize;a++) {
            if (pop[a].hasInflections(lem) && pop[a].typeGeneralise) {
                lemct+=1;
                inflcts[pop[a].getBest(lem)]+=1;
            }
        }
    } else if (learnType==2) {
        for (int a=0;a<runningPopSize;a++) {
            if (pop[a].hasInflections(lem) && !pop[a].typeGeneralise) {
                lemct+=1;
                inflcts[pop[a].getBest(lem)]+=1;
            }
        }
    } else {
        for (int a=0;a<runningPopSize;a++) {
            if (pop[a].hasInflections(lem)) {
                lemct+=1;
                inflcts[pop[a].getBest(lem)]+=1;
            }
        }
    }
    for (int i=0; i<12;i++) {
        inflProbs[i] = float(inflcts[i])/float(lemct);
    }
    
    return getEntropy(inflProbs);
}

//counts high proficiency agents in the population
int highProfCount() {
    int nats = 0;
    for (int i =0; i<runningPopSize; i++) {
        if (pop[i].typeGeneralise) {
            nats+=1;
        }
    }
    return nats;
}

//counts low proficiency agents in the population
int lowProfCount() {
    int nnats = 0;
    for (int i=0; i<runningPopSize;i++) {
        if (!pop[i].typeGeneralise) {
            nnats+=1;
        }
    }
    return nnats;
}


float typesForInfl(int infl, int learnType) {
    int ntypes=0;
    int denom;
    if (learnType==1) {
        denom=highProfCount();
        for (int l = 0; l<28;l++) {
            for (int a=0;a<runningPopSize;a++) {
                if (pop[a].hasInflections(l)) {
                    if (pop[a].getBest(l)== infl) {
                        ntypes+=1;
                    }
                }
            }
        }
    } else if (learnType==2) {
        denom=lowProfCount();
        for (int l = 0; l<28;l++) {
            for (int a=0;a<runningPopSize;a++) {
                if (pop[a].hasInflections(l)) {
                    if (pop[a].getBest(l)== infl) {
                        ntypes+=1;
                    }
                }
            }
        }
    } else {
        denom=runningPopSize;
        for (int l = 0; l<28;l++) {
            for (int a=0;a<runningPopSize;a++) {
                if (pop[a].hasInflections(l)) {
                    if (pop[a].getBest(l)== infl) {
                        ntypes+=1;
                    }
                }
            }
        }
    }
    
    float res = float(ntypes)/float(denom);
    return res;
}



int regRank(int infl, int learnType) {
    //get rank of this inflection among all others
    //sort indices in order of number of types
    //return index of infl (which is itself an index)
    //this rank is necessary to make generalisations across runs, since the identity of the type-dominant inflection will change from run to run
    std::map<int,int> infranks;
    
    
    int infTCounts [] = {0,0,0,0,0,0,0,0,0,0,0,0};//set number of types with each inflection at 0
    if (learnType==1) {
        for (int l = 0; l<28;l++) {//for each lemma in the vocab
            for (int a=0;a<runningPopSize;a++) {//go through each agent
                if (pop[a].hasInflections(l) && pop[a].typeGeneralise) {//if they have inflections for this lemma
                    int best = pop[a].getBest(l);//take the top one
                    infTCounts[best]+=1;
                }
            }
        }
    } else if (learnType==2) {
        for (int l = 0; l<28;l++) {//for each lemma in the vocab
            for (int a=0;a<runningPopSize;a++) {//go through each agent
                if (pop[a].hasInflections(l) && !pop[a].typeGeneralise) {//if they have inflections for this lemma
                    int best = pop[a].getBest(l);//take the top one
                    infTCounts[best]+=1;
                }
            }
        }
    } else {
        for (int l = 0; l<28;l++) {//for each lemma in the vocab
            for (int a=0;a<runningPopSize;a++) {//go through each agent
                if (pop[a].hasInflections(l)) {//if they have inflections for this lemma
                    int best = pop[a].getBest(l);//take the top one
                    infTCounts[best]+=1;
                }
            }
        }
    }
    
    for (int k=0;k<12;k++) {
        infranks[infTCounts[k]] = k;
    }
    //create map with number of types (max=N*vocSize) as key, index of infl as value
    //will automatically be sorted by key in ascending order
    //reverse iterate, add to array by value
    
    int sarray [] = {0,0,0,0,0,0,0,0,0,0,0,0};
    int ind = 0;
    for (auto rit = infranks.rbegin(); rit != infranks.rend(); ++rit) {
        sarray[ind] = rit->second;
        ind+=1;
    }
    //this will make sarray where first element is index of highest ranked inflection

    int r;
    r=0;//give r a default value
    for (int i=0;i<12;i++) {
        if (sarray[i] == infl) {
            r=i;
            break;
        }
    }
    return r+1;
}


//gives the number of types an inflection is the best weight for
int ctTypes(int infl, int learnType) {
    
    int nTypes= 0;
    if (learnType==1) {
        for (int l = 0; l<28;l++) {//for each lemma in the vocab
            for (int a=0;a<runningPopSize;a++) {//go through each agent
                if (pop[a].hasInflections(l) && pop[a].typeGeneralise) {
                    if (pop[a].getBest(l) ==infl) {
                        nTypes+=1;
                    }
                }
            }
        }
    } else if (learnType==2) {
        for (int l = 0; l<28;l++) {//for each lemma in the vocab
            for (int a=0;a<runningPopSize;a++) {//go through each agent
                if (pop[a].hasInflections(l) && !pop[a].typeGeneralise) {
                    if (pop[a].getBest(l) ==infl) {
                        nTypes+=1;
                    }
                }
            }
        }
    } else {
        for (int l = 0; l<28;l++) {//for each lemma in the vocab
            for (int a=0;a<runningPopSize;a++) {//go through each agent
                if (pop[a].hasInflections(l)) {
                    if (pop[a].getBest(l) ==infl) {
                        nTypes+=1;
                    }
                }
            }
        }
    }
    
    return nTypes;
}





int getTopInfl(int lem, int learnType) {
    int tops [] = {0,0,0,0,0,0,0,0,0,0,0,0};;
    
    if (learnType==1) {
        for (int a=0;a<runningPopSize;a++) {
            if (pop[a].hasInflections(lem) && pop[a].typeGeneralise) {
                tops[pop[a].getBest(lem)]+=1;
            }
        }
    } else  if (learnType==2) {
        for (int a=0;a<runningPopSize;a++) {
            if (pop[a].hasInflections(lem) && !pop[a].typeGeneralise) {
                tops[pop[a].getBest(lem)]+=1;
            }
        }
    } else {
        for (int a=0;a<runningPopSize;a++) {
            if (pop[a].hasInflections(lem)) {
                tops[pop[a].getBest(lem)]+=1;
            }
        }
    }
    
    return static_cast<unsigned int>(std::distance(tops,std::max_element(tops,tops+12)));
}


void timeStep(int tNow) {
    
    for (int q = 0; q<popSize; q++) {

        int s = int(round(dis(gen)*runningPopSize-1));

        int h = int(round(dis(gen)*runningPopSize-1));
        while(s==h){
            h = int(round(dis(gen)*runningPopSize-1));
        }

        if (top>=499) {
            //std::cout<<"Shuflfling voclist..."<<std::endl;
            std::random_shuffle(&vocList[0],&vocList[500]);
            top=0;
        }
        
        int topic=vocList[top];

        interaction(s,h,topic,tNow);

        if (growth) {
            addAgent();
        }
        if (replacement) {

            replaceAgent();

        }

        globCounts[topic]+=1;
        allTokens+=1;
        top+=1;
        //std::cout<<"Interaction done.";
        //std::cout<<std::endl;
    }
    
    
}


void singleRun(int runNumber) {
    std::ofstream foutWhole;
    std::ofstream foutByInfl;
    std::ofstream foutEnd;
    //if this is the first run, create the output files
    if (runNumber == 0) {
        foutWhole.open("wholeVocStatic.csv",std::ios_base::app);
        foutWhole<<"Nat\tEntropy\tRun\tTime\tNumInfls\n";
        foutWhole.close();
        
        foutByInfl.open("ByInflStatic.csv",std::ios_base::app);
        foutByInfl<<"Nat\tNumAgents\tInflection\tInflRank\tPropTypes\tCountTypes\tNumTokens\tTotTokens\tTime\tRun\n";
        foutByInfl.close();
        
        foutEnd.open("ByLemEnd.csv",std::ios_base::app);
        foutEnd<<"Nat\tRun\tLemmaIndex\tLemEntropy\tLemCount\tTotTokens\tTopInflRank\tTopInflIndex\n";
        foutEnd.close();
    }
    //if not, add to the output files
    for (int j = 0; j<popSteps; j++) {
        timeStep(j);
        foutWhole.open("wholeVocStatic.csv", std::ios_base::app);
        foutWhole<<"ProficientHigh\t"<<vocabEntropy(1)<<"\t"<<runNumber<<"\t"<<j<<"\t"<<inflsInVoc(1)<<"\n";
        foutWhole<<"ProficientLow\t"<<vocabEntropy(2)<<"\t"<<runNumber<<"\t"<<j<<"\t"<<inflsInVoc(2)<<"\n";
        foutWhole<<"All\t"<<vocabEntropy(3)<<"\t"<<runNumber<<"\t"<<j<<"\t"<<inflsInVoc(3)<<"\n";        foutWhole.close();
        for (int i=0; i<12; i++) {
            if (ctTypes(i,3)>0) {
                foutByInfl.open("ByInflStatic.csv",std::ios_base::app);
                foutByInfl<<"Native\t"<<highProfCount()<<"\t"<<i<<"\t"<<regRank(i,1)<<"\t"<<typesForInfl(i,1)<<"\t"<<ctTypes(i,1)<<"\t"<<globInfls[i]<<"\t"<<allTokens<<"\t"<<j<<"\t"<<runNumber<<"\n";
                foutByInfl<<"Native\t"<<lowProfCount()<<"\t"<<i<<"\t"<<regRank(i,2)<<"\t"<<typesForInfl(i,2)<<"\t"<<ctTypes(i,2)<<"\t"<<globInfls[i]<<"\t"<<allTokens<<"\t"<<j<<"\t"<<runNumber<<"\n";
                foutByInfl<<"All\t"<<runningPopSize<<"\t"<<i<<"\t"<<regRank(i,3)<<"\t"<<typesForInfl(i,3)<<"\t"<<ctTypes(i,3)<<"\t"<<globInfls[i]<<"\t"<<allTokens<<"\t"<<j<<"\t"<<runNumber<<"\n";
                foutByInfl.close();
            }
        }
        
        
    }
    
    std::cout<<"Ending population size: ";
    std::cout<<runningPopSize;
    std::cout<<std::endl;
    
    //after all time steps, give a summary by lemma
    for (int l=0;l<28;l++) {
        
        int topInflNat=getTopInfl(l,1);
        int topInflNon = getTopInfl(l,2);
        int topInfl = getTopInfl(l,3);
        foutEnd.open("ByLemEnd.csv",std::ios::app);
        foutEnd<<"Native\t"<<runNumber<<"\t"<<l<<"\t"<<meaningEntropy(l,1)<<"\t"<<globCounts[l]<<"\t"<<allTokens<<"\t"<<regRank(topInfl,1)<<"\t"<<topInflNat<<"\n";
        foutEnd<<"NonNative\t"<<runNumber<<"\t"<<l<<"\t"<<meaningEntropy(l,2)<<"\t"<<globCounts[l]<<"\t"<<allTokens<<"\t"<<regRank(topInfl,2)<<"\t"<<topInflNon<<"\n";
        foutEnd<<"All\t"<<runNumber<<"\t"<<l<<"\t"<<meaningEntropy(l,3)<<"\t"<<globCounts[l]<<"\t"<<allTokens<<"\t"<<regRank(topInfl,3)<<"\t"<<topInfl<<"\n";
        foutEnd.close();
    }
    
    
}




