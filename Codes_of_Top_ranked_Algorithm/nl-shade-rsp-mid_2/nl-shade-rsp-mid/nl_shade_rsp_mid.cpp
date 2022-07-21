// This is the code of nl_shade_rsp_mid.
// Most of the code is taken from nl_shade_rsp, which was authored by:
// Vladimir Stanovov, Shakhnaz  Akhmedova, Eugene Semenkin 
// and downloaded from: https://github.com/P-N-Suganthan
// Changes were conceptually described in the article: "A Version of NL-SHADE-RSP Algorithm with Midpoint for CEC 2022 Single Objective Bound Constrained Problems"
// In the code, they can be found in conditional compilation blocks enabled by:
// #define K_MEANS_AS_NEAREST
// #define RESAMPLING
// #define COUNT_LIMITS
// Author of the changes: Rafal Biedrzycki

// To compile: (tested under Ubuntu 22.04)
// Use system package manager to install libraries: mlpack and boost (libmlpack-dev, libboost-all-dev)
// compile & link by command: g++ -std=c++17 nl_shade_rsp_mid.cpp -lmlpack -fopenmp

// Before running the program, run command (or-else program will use many threads wchich slowes down in my case):
// export OMP_NUM_THREADS=1

#include <cmath>
#include <iostream>
#include <fstream>
#include <random>

#include <boost/math/statistics/linear_regression.hpp>
#include <cstring>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <mlpack/core/cv/metrics/silhouette_score.hpp>

//the code of test functions to be optimized
#include "RB_cec22_test_func.cpp"

using namespace mlpack::kmeans;
using namespace mlpack::cv;

using namespace std;
using boost::math::statistics::simple_ordinary_least_squares_with_R_squared;


#define K_MEANS_AS_NEAREST
#define RESAMPLING
#define COUNT_LIMITS


unsigned seed1 = 0;
std::mt19937 generator(seed1);
std::uniform_int_distribution<int> uni_int(0,32768);
std::uniform_real_distribution<double> uni_real(0.0,1.0);
std::normal_distribution<double> norm_dist(0.0,1.0);
std::cauchy_distribution<double> cachy_dist(0.0,1.0);

const double MIN_ERROR=1e-8;

int intRandom(int target)
{
    if(target == 0)
        return 0;
    return uni_int(generator)%target;
}
double Random(double minimal, double maximal){return uni_real(generator)*(maximal-minimal)+minimal;}
double NormRand(double mu, double sigma){return norm_dist(generator)*sigma + mu;}
double CachyRand(double mu, double sigma){return cachy_dist(generator)*sigma+mu;}

void seqDouble(vector<double>& seq, int size){
  for(int i=1;i<=size;++i){
    seq[i-1] = i;
  }
}

void getMean(double** popul, int popul_size, int dimensionality, double* mean_indiv){
  for(int dim=0;dim<dimensionality;++dim){
    double sum=0;
    for(int indIndx=0;indIndx<popul_size;++indIndx){
      sum+=popul[indIndx][dim];
    }
    mean_indiv[dim] = sum/popul_size;
  }
}

void getMean(double** popul, int popul_size, int dimensionality, double* mean_indiv, arma::uvec ids){
    int idsSize=ids.size();
  for(int dim=0;dim<dimensionality;++dim){
    double sum=0;
    for(int idsIndx=0;idsIndx<idsSize;++idsIndx){
        int indIndx = ids[idsIndx];
      sum+=popul[indIndx][dim];
    }
    mean_indiv[dim] = sum/idsSize;
  }
}

int getIndxOfNearest(double** popul, int popul_size, int dimensionality, double* mean_indiv){
    int nearestIndx=0;
    double minDist = 1e20;
    for(int indIndx=0;indIndx<popul_size;++indIndx){
        double sqDist=0;
        for(int dim=0;dim<dimensionality;++dim){
            sqDist+= pow(mean_indiv[dim]-popul[indIndx][dim],2);
        }
        if(sqDist<minDist){
            minDist=sqDist;
            nearestIndx = indIndx;
        }
    }
    return nearestIndx;
}

double getDist(double* ind1, double* ind2, int dimensionality){
    double dist=0.0;
    for(int dim=0;dim<dimensionality;++dim){
        dist+= pow(ind1[dim]-ind2[dim], 2);
    }
    return sqrt(dist);
}

void qSort1(double* mass, int low, int high)
{
    int i=low;
    int j=high;
    double x=mass[(low+high)>>1];
    do
    {
        while(mass[i]<x)    ++i;
        while(mass[j]>x)    --j;
        if(i<=j)
        {
            double temp=mass[i];
            mass[i]=mass[j];
            mass[j]=temp;
            i++;    j--;
        }
    } while(i<=j);
    if(low<j)   qSort1(mass,low,j);
    if(i<high)  qSort1(mass,i,high);
}

void qSort2int(double* mass, int* mass2, int low, int high)
{
    int i=low;
    int j=high;
    double x=mass[(low+high)>>1];
    do
    {
        while(mass[i]<x)    ++i;
        while(mass[j]>x)    --j;
        if(i<=j)
        {
            double temp=mass[i];
            mass[i]=mass[j];
            mass[j]=temp;
            int temp2=mass2[i];
            mass2[i]=mass2[j];
            mass2[j]=temp2;
            i++;    j--;
        }
    } while(i<=j);
    if(low<j)
        qSort2int(mass,mass2,low,j);
    if(i<high)
        qSort2int(mass,mass2,i,high);
}

void cec22_test_func(double *, double *,int,int,int);
double *OShift,*M,*y,*z,*x_bound;
int ini_flag=0,n_flag,func_flag,*SS;

const int dimensionality=10;
//const int dimensionality=20;

const int NMBR_OF_RUNS=30;
const int NUM_OF_DUMPS=16;
double stepsFEval[NUM_OF_DUMPS];
double ResultsArray[12][NMBR_OF_RUNS][NUM_OF_DUMPS+1];
int LastFEcount;
int NFEval = 0;
int maxFES = 0;
int evalsAtStart=0;
double tempF[1];
double fopt;
char buffer[200];
double globalBestFit;
bool globalBestFitinit;
bool isRestart=false;

double bestSol[dimensionality];
bool initfinished;
vector<double> FitTemp3;
int foundAt=-1;

void GenerateNextRandUnif(const int num, const int Range, int* Rands, const int Prohib)
{
    for(int j=0;j!=25;j++)
    {
        bool generateagain = false;
        Rands[num] = intRandom(Range);
        for(int i=0;i!=num;i++)
            if(Rands[i] == Rands[num])
                generateagain = true;
        if(!generateagain)
            break;
    }
}

void GenerateNextRandUnifOnlyArch(const int num, const int Range, const int Range2, int* Rands, const int Prohib)
{
    for(int j=0;j!=25;j++)
    {
        bool generateagain = false;
        Rands[num] = intRandom(Range2)+Range;
        for(int i=0;i!=num;i++)
            if(Rands[i] == Rands[num])
                generateagain = true;
        if(!generateagain)
            break;
    }
}

bool CheckGenerated(const int num, int* Rands, const int Prohib)
{
    if(Rands[num] == Prohib)
        return false;
    for(int j=0;j!=num;j++)
        if(Rands[j] == Rands[num])
            return false;
    return true;
}

void SaveBestValues(int funcN, int runNum)
{
    for(int stepFEcount=LastFEcount;stepFEcount<NUM_OF_DUMPS;stepFEcount++)
    {
        if(NFEval == int(stepsFEval[stepFEcount]*maxFES))
        {
            double temp = globalBestFit - fopt;
            if(temp <= MIN_ERROR)
                temp = 0;
            ResultsArray[funcN-1][runNum][stepFEcount] = temp;
            LastFEcount = stepFEcount;
        }
    }
    
}

void FindLimits(double* Ind, double* Parent, int CurNVars, double CurLeft, double CurRight){
    for (int j = 0; j<CurNVars; ++j){
        if (Ind[j] < CurLeft)
            Ind[j] = Random(CurLeft,CurRight);
        if (Ind[j] > CurRight)
            Ind[j] = Random(CurLeft,CurRight);
    }
}

void countLimits(double* ind, int* indLimCounter, int dimensionality, double lower, double upper){
    bool onBound=false;
    for (int dim = 0; dim<dimensionality; ++dim){
        if (ind[dim] <= lower){
            onBound=true;
            break;
        }
        if (ind[dim] >= upper){
            onBound=true;
            break;
        }
    }
    if(onBound){
        ++(*indLimCounter);
    }else{
        *indLimCounter=0;
    }
}

int getNumOfIndOnBounds(double** populTemp, int DIM, int popSize, double lower, double upper){
    int numOfIndOnB=0;
    for (int indIndx = 0; indIndx<popSize; ++indIndx){
        for (int dim = 0; dim<dimensionality; ++dim){
            if( populTemp[indIndx][dim] <= lower || populTemp[indIndx][dim] >= upper){
                ++numOfIndOnB;
                break;
            }
        }
    }
    return numOfIndOnB;
}

bool IsInfeasible(double* Ind, int CurNVars,double CurLeft, double CurRight){
    for (int j = 0; j<CurNVars; ++j){
        if (Ind[j] < CurLeft)
            return true;
        if (Ind[j] > CurRight)
            return true;
    }
    return false;
}

class Optimizer
{
public:
    bool FitNotCalculated;
    int Int_ArchiveSizeParam;
    int MemorySize;
    int MemoryIter;
    int SuccessFilled;
    int MemoryCurrentIndex;
    int NVars;
    int popSize;
    int NIndsMax;
    int NIndsMin;
    int besti;
    int func_num;
    int runNum;
    int Generation;
    int ArchiveSize;
    int CurrentArchiveSize;
    double F;
    double Cr;
    double bestfit;
    double ArchiveSizeParam;
    double Right;
    double Left;
    int* Rands;
    int* Indexes;
    int* BackIndexes;
    double* Weights;
    double* Donor;
    double* Trial;
    double* Fitmass;
    double* popFitTmp;
    double* FitmassCopy;
    double* BestInd;
    double* tempSuccessCr;
    double* tempSuccessF;
    double* FGenerated;
    double* CrGenerated;
    double* MemoryCr;
    double* MemoryF;
    double* FitDelta;
    double* ArchUsages;
    double** Popul;
    double** populTemp;
    double** Archive;
    ofstream infoLog;

    void Initialize(int newNInds, int newNVars, int new_func_num, int newrunNum, int NewMemSize, double NewArchSizeParam);
    void restart(int newNInds, int newNVars, int new_func_num, int newrunNum, int NewMemSize, double NewArchSizeParam);
    void Clean();
    void MainCycle(double);
    void FindNSaveBest(bool init, int ChosenOne);
    inline double GetValue(const int index, const int popSize, const int j);
    void CopyToArchive(double* RefusedParent);
    void SaveSuccessCrF(double Cr, double F, double FitD);
    void UpdateMemoryCrF();
    double MeanWL_general(double* Vector, double* TempWeights, int Size, double g_p, double g_m);
    void RemoveWorst(int popSize, int NewNInds);
    void RemoveTooNear(int popSize, int NewNInds);
};

double cec_22_(double* HostVector,int func_num)
{
	int nx = dimensionality;	
	int mx = 1;
	
	cec22_test_func(HostVector, tempF, nx, mx,func_num);
    NFEval++;
    return tempF[0];
}

void Optimizer::Initialize(int newNInds, int newNVars, int new_func_num, int newrunNum,
                           int NewMemSize, double NewArchSizeParam)
{
    evalsAtStart=0;
    FitNotCalculated = true;
    popSize = newNInds;
    NIndsMax = popSize;
    NIndsMin = 4;
    NVars = newNVars;
    runNum = newrunNum;
    Left = -100;//+10000;
    Right = 100;//+10000;
    Cr = 0.2;
    F = 0.2;
    besti = 0;
    Generation = 0;
    CurrentArchiveSize = 0;
    ArchiveSizeParam = NewArchSizeParam;
    Int_ArchiveSizeParam = ceil(ArchiveSizeParam);
    ArchiveSize = NIndsMax*ArchiveSizeParam;
    func_num = new_func_num;

    for(int steps_k=0;steps_k!=NUM_OF_DUMPS-1;steps_k++)
        stepsFEval[steps_k] = pow(double(dimensionality),double(steps_k)/5.0-3.0);
    stepsFEval[NUM_OF_DUMPS-1] = 1.0;
    Popul = new double*[NIndsMax];
    for(int i=0;i!=NIndsMax;i++)
        Popul[i] = new double[NVars];
    populTemp = new double*[NIndsMax];
    for(int i=0;i!=NIndsMax;i++)
        populTemp[i] = new double[NVars];
    Archive = new double*[NIndsMax*Int_ArchiveSizeParam];
    for(int i=0;i!=NIndsMax*Int_ArchiveSizeParam;i++)
        Archive[i] = new double[NVars];
    Fitmass = new double[NIndsMax];
    popFitTmp = new double[NIndsMax];
    FitmassCopy = new double[NIndsMax];
    Indexes = new int[NIndsMax];
    BackIndexes = new int[NIndsMax];
    BestInd = new double[NVars];
	for (int i = 0; i<NIndsMax; i++)
		for (int j = 0; j<NVars; j++)
			Popul[i][j] = Random(Left,Right);
    Donor = new double[NVars];
    Trial = new double[NVars];
    Rands = new int[NIndsMax];
    tempSuccessCr = new double[NIndsMax];
    tempSuccessF = new double[NIndsMax];
    FitDelta = new double[NIndsMax];
    FGenerated = new double[NIndsMax];
    CrGenerated = new double[NIndsMax];
    for(int i=0;i!=NIndsMax;i++)
    {
        tempSuccessCr[i] = 0;
        tempSuccessF[i] = 0;
    }
    MemorySize = NewMemSize;
    MemoryIter = 0;
    SuccessFilled = 0;
    ArchUsages = new double[NIndsMax];
    Weights = new double[NIndsMax];
    MemoryCr = new double[MemorySize];
    MemoryF = new double[MemorySize];
    for(int i=0;i!=MemorySize;i++)
    {
        MemoryCr[i] = 0.2;
        MemoryF[i] = 0.2;
    }
    foundAt=maxFES;

}

void Optimizer::restart(int newNInds, int newNVars, int new_func_num, int newrunNum,
                           int NewMemSize, double NewArchSizeParam)
{
    evalsAtStart = NFEval;
    FitNotCalculated = true;
    popSize = newNInds;
    NIndsMax = popSize;
    NIndsMin = 4;
    NVars = newNVars;
    runNum = newrunNum;
    Cr = 0.2;
    F = 0.2;
    besti = 0;
    Generation = 0;
    CurrentArchiveSize = 0;
    ArchiveSizeParam = NewArchSizeParam;
    Int_ArchiveSizeParam = ceil(ArchiveSizeParam);
    ArchiveSize = NIndsMax*ArchiveSizeParam;
    func_num = new_func_num;

    Popul = new double*[NIndsMax];
    for(int i=0;i!=NIndsMax;i++)
        Popul[i] = new double[NVars];
    populTemp = new double*[NIndsMax];
    for(int i=0;i!=NIndsMax;i++)
        populTemp[i] = new double[NVars];
    Archive = new double*[NIndsMax*Int_ArchiveSizeParam];
    for(int i=0;i!=NIndsMax*Int_ArchiveSizeParam;i++)
        Archive[i] = new double[NVars];
    Fitmass = new double[NIndsMax];
    popFitTmp = new double[NIndsMax];
    FitmassCopy = new double[NIndsMax];
    Indexes = new int[NIndsMax];
    BackIndexes = new int[NIndsMax];
    BestInd = new double[NVars];

    
	for (int i = 0; i<NIndsMax; i++)
		for (int j = 0; j<NVars; j++)
			Popul[i][j] = Random(Left,Right);
            


    Rands = new int[NIndsMax];
    tempSuccessCr = new double[NIndsMax];
    tempSuccessF = new double[NIndsMax];
    FitDelta = new double[NIndsMax];
    FGenerated = new double[NIndsMax];
    CrGenerated = new double[NIndsMax];
    for(int i=0;i!=NIndsMax;i++)
    {
        tempSuccessCr[i] = 0;
        tempSuccessF[i] = 0;
    }
    MemorySize = NewMemSize;
    MemoryIter = 0;
    SuccessFilled = 0;
    ArchUsages = new double[NIndsMax];
    Weights = new double[NIndsMax];
    MemoryCr = new double[MemorySize];
    MemoryF = new double[MemorySize];
    for(int i=0;i!=MemorySize;i++)
    {
        MemoryCr[i] = 0.2;
        MemoryF[i] = 0.2;
    }


}

void Optimizer::SaveSuccessCrF(double Cr, double F, double FitD)
{
    tempSuccessCr[SuccessFilled] = Cr;
    tempSuccessF[SuccessFilled] = F;
    FitDelta[SuccessFilled] = FitD;
    SuccessFilled ++ ;
}
void Optimizer::UpdateMemoryCrF()
{
    if(SuccessFilled != 0)
    {
        MemoryCr[MemoryIter] =MeanWL_general(tempSuccessCr,FitDelta,SuccessFilled,2,1);
        MemoryF[MemoryIter] = MeanWL_general(tempSuccessF, FitDelta,SuccessFilled,2,1);
        MemoryIter++;
        if(MemoryIter >= MemorySize)
            MemoryIter = 0;
    }
    else
    {
        MemoryF[MemoryIter] = 0.5;
        MemoryCr[MemoryIter] = 0.5;
    }
}
double Optimizer::MeanWL_general(double* Vector, double* TempWeights, int Size, double g_p, double g_m)
{
    double SumWeight = 0;
    double SumSquare = 0;
    double Sum = 0;
    for(int i=0;i!=SuccessFilled;i++)
        SumWeight += TempWeights[i];
    for(int i=0;i!=SuccessFilled;i++)
        Weights[i] = TempWeights[i]/SumWeight;
    for(int i=0;i!=SuccessFilled;i++)
        SumSquare += Weights[i]*pow(Vector[i],g_p);
    for(int i=0;i!=SuccessFilled;i++)
        Sum += Weights[i]*pow(Vector[i],g_p-g_m);
    if(fabs(Sum) > 0.000001)
        return SumSquare/Sum;
    else
        return 0.5;
}

void Optimizer::CopyToArchive(double* RefusedParent)
{
    if(CurrentArchiveSize < ArchiveSize)
    {
        for(int i=0;i!=NVars;i++)
            Archive[CurrentArchiveSize][i] = RefusedParent[i];
        CurrentArchiveSize++;
    }
    else if(ArchiveSize > 0)
    {
        int RandomNum = intRandom(ArchiveSize);
        for(int i=0;i!=NVars;i++)
            Archive[RandomNum][i] = RefusedParent[i];
    }
}

void Optimizer::FindNSaveBest(bool init, int ChosenOne)
{
    if(Fitmass[ChosenOne] <= bestfit || init)
    {
        bestfit = Fitmass[ChosenOne];
        besti = ChosenOne;
        for(int j=0;j!=NVars;j++)
            BestInd[j] = Popul[besti][j];
    }
    if(bestfit < globalBestFit){
        globalBestFit = bestfit;
        memcpy(bestSol, Popul[ChosenOne], dimensionality*sizeof(double));
        }
}

void Optimizer::RemoveTooNear(int popSize, int NewNInds){
    int PointsToRemove = popSize - NewNInds;
    int curPopSize=popSize;
    
    double TOO_SMALL_DIST=1e-7;
    

    for(int L=0;L!=PointsToRemove;L++){
        double smallestDist=1e20;
        int best_i=0;
        int best_j=0;
        for(int i=0;i<curPopSize-1;++i){
            for(int j=i+1;j<curPopSize;++j){
                double dist= getDist(Popul[i], Popul[j], dimensionality);
                if(dist<smallestDist){
                    smallestDist=dist;
                    best_i=i;
                    best_j=j;
                }
                
            }
        }
        if(smallestDist<=TOO_SMALL_DIST){
            for(int j=0;j!=NVars;++j)
                Popul[best_i][j] = Popul[curPopSize-1][j];
            Fitmass[best_i] = Fitmass[curPopSize-1];                
            --curPopSize;
        }

    }
    int pointsLeftToRemove = curPopSize - NewNInds;
    if(pointsLeftToRemove>0){
        RemoveWorst(curPopSize, NewNInds);
    }
}

void Optimizer::RemoveWorst(int popSize, int NewNInds)
{
    int PointsToRemove = popSize - NewNInds;
    for(int L=0;L!=PointsToRemove;L++)
    {
        double WorstFit = Fitmass[0];
        int WorstNum = 0;
        for(int i=1;i!=popSize;i++)
        {
            if(Fitmass[i] > WorstFit)
            {
                WorstFit = Fitmass[i];
                WorstNum = i;
            }
        }
        for(int i=WorstNum;i!=popSize-1;i++)
        {
            for(int j=0;j!=NVars;j++)
                Popul[i][j] = Popul[i+1][j];
            Fitmass[i] = Fitmass[i+1];
        }
    }
}

inline double Optimizer::GetValue(const int index, const int popSize, const int j)
{
    if(index < popSize)
        return Popul[index][j];
    return Archive[index-popSize][j];
}


void dumpInfo(ofstream& infoDump, int popSize, int Generation){
    infoDump<<popSize<<","<<Generation<<endl;
}

void dumpInd(ofstream& popDump, double* ind, const int dimensionality, double fit){
    for(int i=0;i<dimensionality; ++i){
        popDump<<ind[i]<<",";
    }
    popDump<<fit<<endl;
}

void Optimizer::MainCycle(double optFit)
{

    double ArchSuccess;
    double NoArchSuccess;
    double NArchUsages;
    double ArchProbs = 0.5;
    
    double** centroid_matrix= new double*[popSize];
    for(int i=0;i<popSize;++i){
      centroid_matrix[i]=new double[dimensionality];
    }

    vector<double> lam_seq(popSize);
seqDouble(lam_seq, popSize);    
vector<tuple<double, double, double>> abTab(dimensionality);
vector<double> centroids4dim(popSize);
double* est_m = new double[dimensionality]; 
double* mean_indiv = new double[dimensionality]; 
double* mean_indivCl0 = new double[dimensionality]; 
double* mean_indivCl1 = new double[dimensionality]; 
double* mean_indiv_old = new double[dimensionality]; 
int* populLimCount = new int[popSize];
for(int indIndx=0;indIndx<popSize;++indIndx){
    populLimCount[indIndx]=0;
}
int numOfStagIt=0;

const int ITS_MODULO=17;

double fit_mean_old=1e20;


    for(int curIndx=0;curIndx!=popSize;curIndx++)
    {

        Fitmass[curIndx] = cec_22_(Popul[curIndx],func_num);

        FindNSaveBest(curIndx == 0, curIndx);
        if(!globalBestFitinit || bestfit < globalBestFit)
        {
            globalBestFit = bestfit;
            globalBestFitinit = true;
            memcpy(bestSol, Popul[curIndx], dimensionality*sizeof(double));
        }
        SaveBestValues(func_num, runNum);
    }
    do
    {
        double minfit = Fitmass[0];
        double maxfit = Fitmass[0];
        for(int i=0;i!=popSize;i++)
        {
            FitmassCopy[i] = Fitmass[i];
            Indexes[i] = i;
            if(Fitmass[i] >= maxfit)
                maxfit = Fitmass[i];
            if(Fitmass[i] <= minfit)
                minfit = Fitmass[i];
        }
        if(minfit != maxfit)
            qSort2int(FitmassCopy,Indexes,0,popSize-1);
        for(int i=0;i!=popSize;i++)
            for(int j=0;j!=popSize;j++)
                if(i == Indexes[j])
                {
                    BackIndexes[i] = j;
                    break;
                }
        FitTemp3.resize(popSize);
        for(int i=0;i!=popSize;i++)
            FitTemp3[i] = exp(-double(i)/(double)popSize);
        std::discrete_distribution<int> ComponentSelector3(FitTemp3.begin(),FitTemp3.end());

        int psizeval = max(2.0,popSize*(0.2/(double)(maxFES)*(double)(NFEval)+0.2));

        int CrossExponential = 0;
        if(Random(0,1) < 0.5)
            CrossExponential = 1;
        for(int curIndx=0;curIndx!=popSize;curIndx++)
        {
            MemoryCurrentIndex = intRandom(MemorySize);
            Cr = min(1.0,max(0.0,NormRand(MemoryCr[MemoryCurrentIndex],0.1)));
            do
            {
                F = CachyRand(MemoryF[MemoryCurrentIndex],0.1);
            }
            while(F <= 0);
            FGenerated[curIndx] = min(F,1.0);
            CrGenerated[curIndx] = Cr;
        }
        qSort1(CrGenerated,0,popSize-1);
        double iterBestFit = 1E18;
        int iterBestIndx=0;
              
        for(int curIndx=0;curIndx!=popSize;curIndx++)
        {
            Rands[0] = Indexes[intRandom(psizeval)];
            for(int i=0;i!=25 && !CheckGenerated(0,Rands,curIndx);i++)
                Rands[0] = Indexes[intRandom(psizeval)];
            GenerateNextRandUnif(1,popSize,Rands,curIndx);
            if(Random(0,1) > ArchProbs || CurrentArchiveSize == 0)
            {
                Rands[2] = Indexes[ComponentSelector3(generator)];
                for(int i=0;i!=25 && !CheckGenerated(2,Rands,curIndx);i++)
                    Rands[2] = Indexes[ComponentSelector3(generator)];
                ArchUsages[curIndx] = 0;
            }
            else
            {
                GenerateNextRandUnifOnlyArch(2,popSize,CurrentArchiveSize,Rands,curIndx);
                ArchUsages[curIndx] = 1;
            }
            for(int j=0;j!=NVars;j++)
                Donor[j] = Popul[curIndx][j] +
                    FGenerated[curIndx]*(GetValue(Rands[0],popSize,j) - Popul[curIndx][j]) +
                    FGenerated[curIndx]*(GetValue(Rands[1],popSize,j) - GetValue(Rands[2],popSize,j));

            F = FGenerated[curIndx];//Corrected F
            int WillCrossover = intRandom(NVars);
            Cr = CrGenerated[BackIndexes[curIndx]];
            double CrToUse = 0;

            if(NFEval > 0.5*(maxFES))
                CrToUse = (double(NFEval)/double(maxFES)-0.5)*2;

            if(CrossExponential == 0)
            {
                for(int j=0;j!=NVars;j++)
                {
                    if(Random(0,1) < CrToUse || WillCrossover == j)
                        populTemp[curIndx][j] = Donor[j];
                    else
                        populTemp[curIndx][j] = Popul[curIndx][j];
                }
            }
            else
            {
                int StartLoc = intRandom(NVars);
                int L = StartLoc+1;
                while(Random(0,1) < Cr && L < NVars)
                    L++;
                for(int j=0;j!=NVars;j++)
                    populTemp[curIndx][j] = Popul[curIndx][j];
                for(int j=StartLoc;j!=L;j++)
                    populTemp[curIndx][j] = Donor[j];
            }
#ifdef RESAMPLING            
            const int MAX_NUM_OF_TRIALS=100;
            int num_of_trials=1;
            bool usedRepair=false;
            if(IsInfeasible(populTemp[curIndx],NVars,Left,Right)){
                usedRepair=true;
                const int NUM_OF_TR_WITHOUT_F_HANGE=10;
                do{
                    if(num_of_trials>NUM_OF_TR_WITHOUT_F_HANGE){
                        CrossExponential = 0;
                        if(Random(0,1) < 0.5)
                            CrossExponential = 1;
                
                
                        MemoryCurrentIndex = intRandom(MemorySize);
                        Cr = min(1.0,max(0.0,NormRand(MemoryCr[MemoryCurrentIndex],0.1)));
                        do{
                            F = CachyRand(MemoryF[MemoryCurrentIndex],0.1);
                        }
                        while(F <= 0);
                        FGenerated[curIndx] = min(F,1.0);
                        CrGenerated[curIndx] = Cr;
                    }


                Rands[0] = Indexes[intRandom(psizeval)];
                for(int i=0;i!=25 && !CheckGenerated(0,Rands,curIndx);i++)
                    Rands[0] = Indexes[intRandom(psizeval)];
                GenerateNextRandUnif(1,popSize,Rands,curIndx);
                if(Random(0,1) > ArchProbs || CurrentArchiveSize == 0)
                {
                    Rands[2] = Indexes[ComponentSelector3(generator)];
                    for(int i=0;i!=25 && !CheckGenerated(2,Rands,curIndx);i++)
                        Rands[2] = Indexes[ComponentSelector3(generator)];
                    ArchUsages[curIndx] = 0;
                }
                else
                {
                    GenerateNextRandUnifOnlyArch(2,popSize,CurrentArchiveSize,Rands,curIndx);
                    ArchUsages[curIndx] = 1;
                }
                for(int j=0;j!=NVars;j++)
                    Donor[j] = Popul[curIndx][j] +
                        FGenerated[curIndx]*(GetValue(Rands[0],popSize,j) - Popul[curIndx][j]) +
                        FGenerated[curIndx]*(GetValue(Rands[1],popSize,j) - GetValue(Rands[2],popSize,j));
                F = FGenerated[curIndx]; //corrected F
                int WillCrossover = intRandom(NVars);
                if(num_of_trials<=NUM_OF_TR_WITHOUT_F_HANGE){
                    Cr = CrGenerated[BackIndexes[curIndx]];
                }
                double CrToUse = 0;
                if(NFEval > 0.5*(maxFES))
                    CrToUse = (double(NFEval)/double(maxFES)-0.5)*2;
                if(CrossExponential == 0)
                {
                    for(int j=0;j!=NVars;j++)
                    {
                        if(Random(0,1) < CrToUse || WillCrossover == j)
                            populTemp[curIndx][j] = Donor[j];
                        else
                            populTemp[curIndx][j] = Popul[curIndx][j];
                    }
                }
                else
                {
                    int StartLoc = intRandom(NVars);
                    int L = StartLoc+1;
                    while(Random(0,1) < Cr && L < NVars)
                        L++;
                    for(int j=0;j!=NVars;j++)
                        populTemp[curIndx][j] = Popul[curIndx][j];
                    for(int j=StartLoc;j!=L;j++)
                        populTemp[curIndx][j] = Donor[j];
                }
                ++num_of_trials;
                }while(IsInfeasible(populTemp[curIndx],NVars,Left,Right)&&num_of_trials<=MAX_NUM_OF_TRIALS);
                if(IsInfeasible(populTemp[curIndx],NVars,Left,Right)){
                    usedRepair=false;
                    cout<<"still infeasible!"<<endl;
                    FindLimits(populTemp[curIndx],Popul[curIndx],NVars,Left,Right);

                }
            }


#ifdef COUNT_LIMITS
const int MIN_ITS_ON_BOUND=9;//best

//const int MIN_ITS_ON_BOUND=10;
//const int MIN_ITS_ON_BOUND=11;
//const int MIN_ITS_ON_BOUND=8;
//const int MIN_ITS_ON_BOUND=7;


countLimits(populTemp[curIndx], &(populLimCount[curIndx]), NVars,Left,Right);
if(populLimCount[curIndx]>MIN_ITS_ON_BOUND){
    //cout<<"Bounds restart, curBestSol:"<<globalBestFit-optFit<<" FES:"<<NFEval<<"-----------------------------------------------"<<endl;
    return;
}

#endif  

#endif            

            popFitTmp[curIndx] = cec_22_(populTemp[curIndx],func_num);
            if(popFitTmp[curIndx] < iterBestFit){
              iterBestFit = popFitTmp[curIndx];
              iterBestIndx=curIndx;
            }
            if(popFitTmp[curIndx] <= globalBestFit){
                globalBestFit = popFitTmp[curIndx];
                memcpy(bestSol, populTemp[curIndx], dimensionality*sizeof(double));
                if(globalBestFit-optFit<=MIN_ERROR){
                  foundAt = NFEval;
                  ResultsArray[func_num-1][runNum][NUM_OF_DUMPS] = foundAt;
                  SaveBestValues(func_num,runNum);
                  cout<<"Found Opt----------------------------"<<endl;
                  return;
                }
            }

            if(popFitTmp[curIndx] < Fitmass[curIndx])
                SaveSuccessCrF(Cr,F,fabs(Fitmass[curIndx]-popFitTmp[curIndx]));
            
            FindNSaveBest(false,curIndx);
            SaveBestValues(func_num,runNum);
            
            if(NFEval>maxFES){
                cout<<"Max Evals----------------------------"<<endl;

                return;
            }            
        }
        int CHOSEN_INDX=0;


#ifdef K_MEANS_AS_NEAREST
        arma::mat data(  dimensionality, popSize ); // n_rows, n_cols
        // The assignments will be stored in this vector.
        for(int i=0;i<popSize;++i){
            for(int j=0;j<dimensionality;++j){
                data(j,i) = populTemp[i][j];//(i,j) at the i-th row and j-th column
            }
        }
        arma::Row<size_t> assignments;
        arma::Row<size_t> bestAssignments;
        arma::mat centroids;
        arma::mat bestCentroids;
        
        //const int MAX_K=5;
        const int MAX_K=2;

        //const int MIN_POP_SIZE_4_SPLIT = 4;
        //const int MIN_POP_SIZE_4_SPLIT = 10;
        const int MIN_POP_SIZE_4_SPLIT = 20;
        double bestSilhouette=-1;
        int best_k=2;
        
        mlpack::metric::EuclideanDistance metric;
        for( int cand_k=2; cand_k<=MAX_K; ++cand_k){
            KMeans<> k;
            k.Cluster(data, cand_k, assignments, centroids);    
        
            double silhouetteScore = SilhouetteScore::Overall(data, assignments, metric);
            if(silhouetteScore>bestSilhouette){
                bestSilhouette=silhouetteScore;
                best_k = cand_k;
                bestAssignments = assignments;
                bestCentroids = centroids;
            }
        }
        double bestCandFit=1e20;
        
        double MIN_silhouette = 1/(4*sqrt(dimensionality));
        //double MIN_silhouette = 1/(5*sqrt(dimensionality));
        //double MIN_silhouette = 1/(3*sqrt(dimensionality));
        if(bestSilhouette>MIN_silhouette && popSize>=MIN_POP_SIZE_4_SPLIT){
            for( int cur_k=0; cur_k<best_k; ++cur_k){
                for(int i=0;i<dimensionality;++i){
                    mean_indiv[i] = bestCentroids(i,cur_k);
                    FindLimits(mean_indiv,mean_indiv,NVars,Left,Right);
                }
                double fit_mean = cec_22_(mean_indiv, func_num); 

                if(fit_mean<bestCandFit){
                    bestCandFit=fit_mean;
                    for(int i=0;i<dimensionality;++i){
                        mean_indivCl0[i]=mean_indiv[i];
                    }
                }

                if(fit_mean < globalBestFit){
                    globalBestFit = fit_mean;
                    memcpy(bestSol, mean_indiv, dimensionality*sizeof(double));
                }        
                SaveBestValues(func_num,runNum);
                if(globalBestFit-optFit<=MIN_ERROR){
                    foundAt = NFEval;
                    ResultsArray[func_num-1][runNum][NUM_OF_DUMPS] = foundAt;
                    cout<<"Found Opt Mean----------------------------"<<endl;
                    return;
                }

                //Sale All in pop:
                CHOSEN_INDX=getIndxOfNearest(populTemp, popSize, dimensionality, mean_indiv);
                if( fit_mean<popFitTmp[CHOSEN_INDX] ){
                    popFitTmp[CHOSEN_INDX] = fit_mean;
                    memcpy(populTemp[CHOSEN_INDX], mean_indiv, dimensionality*sizeof(double));
                }


            }//for( int cur_k=0; cur_k<best_k; ++cur_k){


        }else{//kmaeans failed- use just mean
            getMean(populTemp, popSize, dimensionality, mean_indiv);
            FindLimits(mean_indiv,mean_indiv,NVars,Left,Right);
            double fit_mean=cec_22_(mean_indiv,func_num);  
            if(fit_mean < globalBestFit){
                globalBestFit = fit_mean;
                memcpy(bestSol, mean_indiv, dimensionality*sizeof(double));
            }        
            SaveBestValues(func_num,runNum);
            if(globalBestFit-optFit<=MIN_ERROR){
                foundAt = NFEval;
                ResultsArray[func_num-1][runNum][NUM_OF_DUMPS] = foundAt;
                cout<<"Found Opt Mean----------------------------"<<endl;
                return;
            }
            CHOSEN_INDX=getIndxOfNearest(populTemp, popSize, dimensionality, mean_indiv);
            if( fit_mean<popFitTmp[CHOSEN_INDX] ){
                popFitTmp[CHOSEN_INDX] = fit_mean;
                memcpy(populTemp[CHOSEN_INDX], mean_indiv, dimensionality*sizeof(double));
            }


            
        }//else{//kmaeans failed- use just mean

        const double MIN_DIST=1e-9;
        //const int MIN_numOfStagIt=7;//best

        const int MIN_numOfStagIt=8;
        //const int MIN_numOfStagIt=9;
        //const int MIN_numOfStagIt=6;
        //const int MIN_numOfStagIt=5;
        getMean(populTemp, popSize, dimensionality, mean_indiv);
       if(getDist(mean_indiv_old, mean_indiv, dimensionality)<MIN_DIST){
            numOfStagIt+=1;
            if(numOfStagIt>MIN_numOfStagIt){
                cout<<"Restart dist stagIt, curBestSol:"<<globalBestFit-optFit<<" FES:"<<NFEval<<"-----------------------------------------------"<<endl;
                return;
            }
        }else{
            numOfStagIt=0;
        }
        memcpy(mean_indiv_old, mean_indiv, dimensionality*sizeof(double));
        //cout<<endl;

#endif     


        ArchSuccess = 0;
        NoArchSuccess = 0;
        NArchUsages = 0;
        for(int curIndx=0;curIndx!=popSize;curIndx++)
        {
            if(popFitTmp[curIndx] <= Fitmass[curIndx])
            {
                if(ArchUsages[curIndx] == 1)
                {
                    ArchSuccess += (Fitmass[curIndx] - popFitTmp[curIndx])/Fitmass[curIndx];
                    NArchUsages += 1;
                }
                else
                    NoArchSuccess+=(Fitmass[curIndx] - popFitTmp[curIndx])/Fitmass[curIndx];
                CopyToArchive(Popul[curIndx]);
            }
            if(popFitTmp[curIndx] <= Fitmass[curIndx])
            {
                for(int j=0;j!=NVars;j++)
                    Popul[curIndx][j] = populTemp[curIndx][j];
                Fitmass[curIndx] = popFitTmp[curIndx];
            }
        }
        if(NArchUsages != 0)
        {
            ArchSuccess = ArchSuccess/NArchUsages;
            NoArchSuccess = NoArchSuccess/(popSize-NArchUsages);
            ArchProbs = ArchSuccess/(ArchSuccess + NoArchSuccess);
            ArchProbs = max(0.1,min(0.9,ArchProbs));
            if(ArchSuccess == 0)
                ArchProbs = 0.5;
        }
        else
            ArchProbs = 0.5;
        int newNInds;
        if(isRestart){//HomoAtReset
            double shapeConst = 0.1;//jak większe to szybszy spadek, mozna tez pokręcić minimalnym rozmiarem pop
            //double shapeConst = 0.2;//jak większe to szybszy spadek, mozna tez pokręcić minimalnym rozmiarem pop
            //double shapeConst = 0.05;

            //const int MIN_POP=4;

            const int MIN_POP=20;//best

            //const int MIN_POP=30;
            //const int MIN_POP=10;

            double divider = (maxFES-evalsAtStart)/shapeConst;

            double delta=pow( (MIN_POP* (maxFES-evalsAtStart)/divider-(maxFES-evalsAtStart)/divider*popSize), 2)-4*(maxFES-evalsAtStart)/divider*(MIN_POP-popSize);
            if(delta<=0){
                newNInds=MIN_POP;
            }else{
            double b1=(-(MIN_POP*(maxFES-evalsAtStart)/divider-(maxFES-evalsAtStart)/divider*popSize)-sqrt(delta))/(2*(MIN_POP-popSize));
            double a1=popSize-1/b1;
            newNInds = round(a1+1/( (NFEval-evalsAtStart) /divider+b1));
            }
        }else{
            newNInds = round((NIndsMin-NIndsMax)*pow((double(NFEval)/double(maxFES)),(1.0-double(NFEval)/double(maxFES)))+NIndsMax);
        }


        if(newNInds < NIndsMin)
            newNInds = NIndsMin;
        if(newNInds > NIndsMax)
            newNInds = NIndsMax;
        int newArchSize = round((NIndsMin-NIndsMax)*pow((double(NFEval)/double(maxFES)),(1.0-double(NFEval)/double(maxFES)))+NIndsMax)*ArchiveSizeParam;
        if(newArchSize < NIndsMin)
            newArchSize = NIndsMin;
        ArchiveSize = newArchSize;
        if(CurrentArchiveSize >= ArchiveSize)
            CurrentArchiveSize = ArchiveSize;
        RemoveWorst(popSize,newNInds);
        popSize = newNInds;
        UpdateMemoryCrF();
        SuccessFilled = 0;
        Generation ++;
    } while(NFEval < maxFES);
    delete [] est_m;
    delete [] mean_indiv;
}

void Optimizer::Clean()
{
    delete Donor;
    delete Trial;
    delete Rands;
    for(int i=0;i!=NIndsMax;i++)
    {
        delete Popul[i];
        delete populTemp[i];
    }
    for(int i=0;i!=NIndsMax*Int_ArchiveSizeParam;i++)
        delete Archive[i];
    delete ArchUsages;
    delete Archive;
    delete Popul;
    delete populTemp;
    delete Fitmass;
    delete popFitTmp;
    delete FitmassCopy;
    delete BestInd;
    delete Indexes;
    delete BackIndexes;
    delete tempSuccessCr;
    delete tempSuccessF;
    delete FGenerated;
    delete CrGenerated;
    delete FitDelta;
    delete MemoryCr;
    delete MemoryF;
    delete Weights;
}

int main()
{
    float f;
    FILE* rs;
    rs = fopen("input_data/Rand_Seeds.txt","r+");
    int Seeds[1000];
    for(int i=0;i!=1000;i++)
    { 
        fscanf(rs,"%f",&f);
        Seeds[i] = int(f);
    }

    {
        if(dimensionality == 10){
            maxFES = 200000;
        }
        if(dimensionality == 20){
            maxFES = 1000000;
        }
        int POP_SIZE=dimensionality*5;
        

        int oldPopSize=POP_SIZE;


            for (int runNum = 0;runNum!=NMBR_OF_RUNS;runNum++)
            {
                for(int func_num = 1; func_num < 13; func_num++)
                {

                    int seed_ind=(dimensionality/10*func_num*NMBR_OF_RUNS+runNum)-NMBR_OF_RUNS;
                    seed_ind=seed_ind%1000+1;

                    seed1 = Seeds[seed_ind];
                    generator.seed(seed1);
                    fopt = 0;
                        switch(func_num)
                        {
                            case 1: {fopt = 300;  break;}
                            case 2: {fopt = 400; break;}
                            case 3: {fopt = 600;  break;}
                            case 4: {fopt = 800; break;}
                            case 5: {fopt = 900; break;}
                            case 6: {fopt = 1800; break;}
                            case 7: {fopt = 2000; break;}
                            case 8: {fopt = 2200; break;}
                            case 9: {fopt = 2300; break;}
                            case 10:{fopt = 2400; break;}
                            case 11:{fopt = 2600; break;}
                            case 12:{fopt = 2700; break;}
                        }
                    globalBestFitinit = false;
                    initfinished = false;
                    LastFEcount = 0;
                    NFEval = 0;
                    Optimizer OptZ;
                    ////////////////popSize     NVars  func     Run  bench     memory    arch size
 
 #if defined (SR_SAVE_AS_NEAREST_RESTART) || defined (K_MEANS_AS_NEAREST)
                    
                    OptZ.Initialize(POP_SIZE,dimensionality,func_num,runNum,20*dimensionality,2.1);
                    isRestart=false;
                    do{
                        OptZ.MainCycle(fopt);
                        
                        POP_SIZE = 400; //best

                        OptZ.restart(POP_SIZE,dimensionality,func_num,runNum,20*dimensionality,2.1);
                        isRestart=true;

                    }while(globalBestFit-fopt>MIN_ERROR && NFEval<maxFES);
                    POP_SIZE=oldPopSize;
                    isRestart=false;
#endif                    

                    OptZ.Clean();
                    cout<<dimensionality<<" Run "<<runNum+1<<" f "<<func_num<<" global best "<<globalBestFit-fopt<<" best sol:";
                    cout<<endl;
                }
            }

            for(int func_num=1;func_num<13;func_num++)
            {
                string filename = "NL-SHADE-RSP_MID";

                sprintf(buffer,"%d_%d_pop:%d.txt",func_num,dimensionality, POP_SIZE);
                filename = filename + buffer;
                cout<<filename<<endl;
                ofstream fout(filename.c_str(),ios::trunc);
                for(int step = 0;step!=17;step++)
                {
                    for(int runNum = 0;runNum!=NMBR_OF_RUNS;runNum++){
                        if(step==16 && ResultsArray[func_num-1][runNum][step]==0){
                            ResultsArray[func_num-1][runNum][step] = maxFES;
                        }
                        fout<<ResultsArray[func_num-1][runNum][step]<<"\t";
                    }
                        
                    fout<<endl;
                }
                
            }
        
    }
	return 0;
}
