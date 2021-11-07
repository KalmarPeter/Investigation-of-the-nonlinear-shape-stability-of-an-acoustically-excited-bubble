#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>
#define _USE_MATH_DEFINES
#include <math.h>

#define NumberOfModes 9
#define NumberOfIntervals 200
#define Perturbation 1e-7 //1e-4
#define T 1.0
#define AbsTolerance 1.0e-8
#define RelTolerance 1.0e-8
#define SysDim 22
#define SysDimKM 2
#define TransientIterationKM 1024
#define TransientIteration 354
#define ConvergedIteration 32
#define GrowLimit  5.0
#define ShrinkLimit  0.1

// Functions

void Sample(double, double, double*);
void GetRp(double, double*, double, double, double, double, double, double, double, double);
void GetC(double, double, double*, double, double, double, double, double, double, double, double, double, double, double);
void OdeFunKM(double*, double&, double&, double*, double*);
void OdeFun(double*, double&, double&, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*);
void rkckKM(double&, double*, double*, double&, double*, double*);
void rkck(double&, double*, double*, double*, double&, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*);
void GetTolerance(double*, double*, double*);
void GetTimeStep(double*, double*, double*, double*, double&, double&, double&, double*, double*);
void OdeSolverKM(double&, double*, double*, double*, double*, double*);
void OdeSolver(double&, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*);

clock_t SimulationStart = clock();

//Control parameters:

double R0; // equilibrium bouble size
double LowerBoundary = 2.0e-6;
double UpperBoundary = 10.0e-6;
bool AvoidCrash = 0;

// Features of the fluid:

double penv = 1.0e5; // environmental pressure
double pv = 6.11; // vapor pressure
double n = 1.4; // polytropic exponent
double rho = 1574.0; // density
double sft = 0.07987; // surface tension
double nu = 0.015; // viscosity
double cl = 1671.0; // speed of sound in water
double theta = 0.0;

const double fr0 = (1.0 / (2.0 * M_PI * R0 * pow(rho, (1.0 / 2.0)))) * pow((3.0 * n * (penv + (2.0 * sft / R0) - pv) - (2.0 * sft / R0) - (4.0 * pow(nu, 2.0)) / (rho * pow(R0, 2.0))), (1.0 / 2.0)); // own frequency of the bubble
const double omega0 = pow(((3.0 * n * (penv - pv)) / (rho * R0 * R0)) + ((2.0 * (3.0 * n - 1.0) * sft) / (rho * R0)), (1.0 / 2.0)); // own angular speed of the bubble

int main()
{
	// Initial Conditions

	double t0;
	double* y = new double[SysDim];
	double* Rm = new double[SysDim];
	double* Rt = new double[SysDim];

	// Parameters of the excitation

	double pa1 = 5.0e5; // pressure amplitude 1
	double pa2 = 0.0; // pressure amplitude 2
	double fr1 = 26.12e3;
	double fr2 = 0.0;
	double omega1 = 2.0 * M_PI * fr1; // angular speed 1
	double omega2 = 2.0 * M_PI * fr2; // angular speed 2

	// Read matrices which has been created in python

	double* ia = new double[NumberOfModes * NumberOfModes * NumberOfModes];
	double* gd = new double[NumberOfModes * NumberOfModes * NumberOfModes];
	double* ma = new double[NumberOfModes * NumberOfModes * NumberOfModes];
	double* mb = new double[NumberOfModes * NumberOfModes * NumberOfModes];
	double* mc = new double[NumberOfModes * NumberOfModes * NumberOfModes];
	double* md = new double[NumberOfModes * NumberOfModes * NumberOfModes];
	double* qb = new double[NumberOfModes * NumberOfModes * NumberOfModes];
	double* qc = new double[NumberOfModes * NumberOfModes * NumberOfModes];

	std::ifstream Ia;
	Ia.open("ia.txt");

	std::ifstream Gd;
	Gd.open("gd.txt");

	std::ifstream Ma;
	Ma.open("ma.txt");

	std::ifstream Mb;
	Mb.open("mb.txt");

	std::ifstream Mc;
	Mc.open("mc.txt");

	std::ifstream Md;
	Md.open("md.txt");

	std::ifstream Qb;
	Qb.open("qb.txt");

	std::ifstream Qc;
	Qc.open("qc.txt");

	for (int x = 0; x < NumberOfModes * NumberOfModes * NumberOfModes; x++)
	{
		Ia >> ia[x];
		Gd >> gd[x];
		Ma >> ma[x];
		Mb >> mb[x];
		Mc >> mc[x];
		Md >> md[x];
		Qb >> qb[x];
		Qc >> qc[x];
	}

	// Define d_c pointer for GetC function

	double* c = new double[23];
	double* rp = new double[7];
	double* R = new double[NumberOfIntervals];

	// Define R for parameter sweep (radius)

	Sample(LowerBoundary, UpperBoundary, R);

	//////////////////////////////////////////////////////////////////////////////////

	// Create a txt

	std::ofstream DataFile;
	DataFile.open("rkck_cuda.txt");
	int Width = 18;
	DataFile.precision(10);
	DataFile.flags(std::ios::scientific);

	// Main loop:

	for (int r = 0; r < NumberOfIntervals; r++)
	{
		// Initial conditions 

		R0 = R[r];
		t0 = 0.0;
		AvoidCrash = 0;
		for (int i = 0; i < SysDim / 2; i++)
		{
			y[2 * i] = 0.0;
			y[2 * i + 1] = 0.0;
		}
		y[0] = 1.0;

		std::cout << "R: " << R0 << "  Number: " << r << std::endl;
		
		// Get constans for OdeFun

		GetRp(omega1, rp, penv, pv, R0, rho, sft, pa1, pa2, nu);
		GetC(omega1, omega2, c, penv, pv, R0, rho, cl, n, sft, pa1, pa2, nu, theta);

		// Ode Solver

		// Solving the Keller-Miksis eq.
		
		for (int z = 0; z < TransientIterationKM; z++)
		{
			OdeSolverKM(t0, y, c, rp, Rm, Rt);
			//std::cout << "K-M: " << z << std::endl;
			if (AvoidCrash == 1)
			{
				std::cout << " Simulation aborted (KM)" << std::endl;
				DataFile.width(Width); DataFile << R0 << ',';
				DataFile.width(Width); DataFile << " Simulation aborted(KM)";
				DataFile << '\n';
				break;
			}
		}

		if (AvoidCrash == 1)
		{
			continue;
		}
		

		// Perturbation after transient iteration

		for (int i = 2; i < NumberOfModes + 2; i++)
		{
			y[2 * i] = Perturbation;
			y[2 * i + 1] = 0.0;
		}

		// Solving the Keller-Miksis eq. with coupled terms according to shaw

		for (int z = 0; z < TransientIteration; z++)
		{
			OdeSolver(t0, y, c, rp, Rm, Rt, ia, gd, ma, mb, mc, md, qb, qc);
			//std::cout << "CouplingTransient: " << z << std::endl;
			if (AvoidCrash == 1)
			{
				std::cout << " Simulation aborted (Transient coupled KM)" << std::endl;
				DataFile.width(Width); DataFile << R0 << ',';
				DataFile.width(Width); DataFile << " Simulation aborted(Transient coupled KM)";
				DataFile << '\n';
				break;
			}
		}

		if (AvoidCrash == 1)
		{
			continue;
		}

		for (int i = 0; i < SysDim; i++)
		{
			Rm[i] = 0.0;
			Rt[i] = 0.0;
		}

		for (int z = 0; z < ConvergedIteration; z++)
		{
			OdeSolver(t0, y, c, rp, Rm, Rt, ia, gd, ma, mb, mc, md, qb, qc);
			//std::cout << "CouplingConverged: " << z << std::endl;
			if (AvoidCrash == 1)
			{
				std::cout << " Simulation aborted (Converged coupled KM)" << std::endl;
				DataFile.width(Width); DataFile << R0 << ',';
				DataFile.width(Width); DataFile << " Simulation aborted(Converged coupled KM)";
				DataFile << '\n';
				break;
			}

			DataFile.width(Width); DataFile << Rm[0] << ',';
			DataFile.width(Width); DataFile << Rm[2] << ',';
			DataFile.width(Width); DataFile << Rm[4] << ',';
			DataFile.width(Width); DataFile << Rm[6] << ',';
			DataFile.width(Width); DataFile << Rm[8] << ',';
			DataFile.width(Width); DataFile << Rm[10] << ',';
			DataFile.width(Width); DataFile << Rm[12] << ',';
			DataFile.width(Width); DataFile << Rm[14] << ',';
			DataFile.width(Width); DataFile << Rm[16] << ',';
			DataFile.width(Width); DataFile << Rm[18] << ',';

			DataFile.width(Width); DataFile << Rt[0] << ',';
			DataFile.width(Width); DataFile << Rt[2] << ',';
			DataFile.width(Width); DataFile << Rt[4] << ',';
			DataFile.width(Width); DataFile << Rt[6] << ',';
			DataFile.width(Width); DataFile << Rt[8] << ',';
			DataFile.width(Width); DataFile << Rt[10] << ',';
			DataFile.width(Width); DataFile << Rt[12] << ',';
			DataFile.width(Width); DataFile << Rt[14] << ',';
			DataFile.width(Width); DataFile << Rt[16] << ',';
			DataFile.width(Width); DataFile << Rt[18] << ',';
			
			DataFile.width(Width); DataFile << R0 << ',';
			DataFile.width(Width); DataFile << t0;
			DataFile << '\n';
		}
	}

	//Write maximum bubble radius at the given sampleing to txt
	DataFile.close();

	clock_t SimulationEnd = clock();
	std::cout << 1000 * (SimulationEnd - SimulationStart) / CLOCKS_PER_SEC << std::endl;
	return 0;
}

// Linear Sampling

void Sample(double LowerBoundary, double UpperBoundary, double* R)
{
	double h = (UpperBoundary - LowerBoundary) / (double(NumberOfIntervals) - 1.0);
	double Point = LowerBoundary - h;

	for (int i = 0; i < NumberOfIntervals; i++)
	{
		R[i] = Point + h;
		Point = R[i];
	}
}


// Constans for dimensionless Ryleigh Plesset equation

void GetRp(double omega1, double* rp, double penv, double pv, double R0, double rho, double sft, double pa1, double pa2, double nu)
{
	double cons = pow(((2.0 * M_PI) / (R0 * omega1)), 2.0);
	rp[0] = ((pv - penv) / rho) * cons;
	rp[1] = (1.0 / rho) * ((2.0 * sft / R0) - pv + penv) * cons;
	rp[2] = 3.0 * n;
	rp[3] = ((2.0 * sft) / (rho * R0)) * cons;
	rp[4] = ((4.0 * nu) / (rho * pow(R0, 2.0))) * ((2.0 * M_PI) / omega1);
	rp[5] = (pa1 / rho) * cons;
	rp[6] = (pa2 / rho) * cons;
}

// Constanc for dimensionless Keller-Miksis eq

void GetC(double omega1, double omega2, double* c, double penv, double pv, double R0, double rho, double cl, double n, double sft, double pa1, double pa2, double nu, double theta)
{

	c[0] = ((4.0 * M_PI * M_PI) / (R0 * R0 * omega1 * omega1 * rho)) * (((2.0 * sft) / (R0)) + penv - pv);
	c[1] = (((1.0 - 3.0 * n) * 2.0 * M_PI) / (R0 * omega1 * rho * cl)) * (((2.0 * sft) / R0) + penv - pv);
	c[2] = ((penv - pv) * 4.0 * M_PI * M_PI) / (R0 * R0 * omega1 * omega1 * rho);
	c[3] = (8.0 * M_PI * M_PI * sft) / (R0 * R0 * R0 * omega1 * omega1 * rho);
	c[4] = (8.0 * M_PI * nu) / (R0 * R0 * omega1 * rho);
	c[5] = (4.0 * M_PI * M_PI * pa1) / (R0 * R0 * omega1 * omega1 * rho);
	c[6] = (4.0 * M_PI * M_PI * pa2) / (R0 * R0 * omega1 * omega1 * rho);
	c[7] = (4.0 * M_PI * M_PI * pa1) / (R0 * omega1 * rho * cl);
	c[8] = (4.0 * M_PI * M_PI * omega2 * pa2) / (R0 * omega1 * omega1 * rho * cl);
	c[9] = (R0 * omega1) / (2.0 * M_PI * cl);
	c[10] = (3.0 * n);
	c[11] = (omega2) / (omega1);
	c[12] = theta;
	c[13] = (R0 * omega1 / (2.0 * M_PI));
	c[14] = ((nu * 2.0 * M_PI) / (rho * R0 * R0 * omega1));
	c[15] = (1.0 / (c[13] * c[13] * rho));
	c[16] = ((4.0 * nu) / (cl * rho * R0));
	c[17] = (c[13] / cl);
	c[18] = ((sft * 4.0 * M_PI * M_PI) / (rho * R0 * R0 * R0 * omega1 * omega1));
	c[19] = penv;
	c[20] = pa1;
	c[21] = pa2;
	c[22] = (2.0 * sft / R0 - (pv - penv));
}

// Domensionless Keller - Miksis equation for trainsient iteration

void OdeFunKM(double* f, double& dt, double& t0, double* y, double* c)
{
	// Keller-Miksis eq.

	double rply;
	double arg1;
	double arg2;

	rply = 1.0 / y[0];
	arg1 = 2.0 * M_PI * t0;
	arg2 = 2.0 * M_PI * c[11] * t0 + c[12];

	double N;
	double D;
	double NPerD;

	f[0] = y[1];

	N = (c[0] + c[1] * y[1]) * pow(rply, c[10]) - c[2] * (1.0 + c[9] * y[1]) - c[3] * rply - c[4] * y[1] * rply
		- (1.5 - 0.5 * c[9] * y[1]) * y[1] * y[1] - (c[5] * sin(arg1) + c[6] * sin(arg2)) * (1.0 + c[9] * y[1])
		- y[0] * (c[7] * cos(arg1) + c[8] * cos(arg2));

	D = (y[0] - c[9] * y[0] * y[1] + c[4] * c[9]);

	NPerD = N / D;

	f[1] = NPerD;
}

// Dimensionless Keller - Miksis equation with coupling terms according to shaw

void OdeFun(double* f, double& dt, double& t0, double* y, double* c, double* rp, double* ia, double* gd, double* ma, double* mb, double* mc, double* md, double* qb, double* qc)
{
	// First derivatives

	y[SysDim - 1] = 0.0;
	y[SysDim - 2] = 0.0;

	for (int i = 0; i < SysDim / 2; i++)
	{
		f[2 * i] = y[2 * i + 1];
	}

	// Dimensionless Rayleight Plesset eq.as a smart guess

	double rply;
	double arg1;
	double arg2;

	rply = 1.0 / y[0];
	arg1 = 2.0 * M_PI * t0;
	arg2 = 2.0 * M_PI * c[11] * t0 + c[12];

	f[1] = rply * (rp[0] + rp[1] * pow(rply, rp[2]) - rp[3] * rply - rp[4] * y[1] * rply - rp[5] * sin(arg1) - rp[6] * sin(arg2) - (3.0 / 2.0) * (y[1] * y[1]));

	// Keller-Miksis eq.

	double N;
	double D;
	//double NPerD;


	N = (c[0] + c[1] * y[1]) * pow(rply, c[10]) - c[2] * (1.0 + c[9] * y[1]) - c[3] * rply - c[4] * y[1] * rply
		- (1.5 - 0.5 * c[9] * y[1]) * y[1] * y[1] - (c[5] * sin(arg1) + c[6] * sin(arg2)) * (1.0 + c[9] * y[1])
		- y[0] * (c[7] * cos(arg1) + c[8] * cos(arg2));

	D = (y[0] - c[9] * y[0] * y[1] + c[4] * c[9]);

	//NPerD = N / D;

	//f[1] = NPerD;

	// Pressure Terms

	double PressureTermVol = (c[19] + c[20] * sin(arg1) + c[21] * sin(arg2) - c[22] * (1.0 - c[10]) * pow(rply, c[10]));
	double PressureTermMod = (c[19] + c[20] * sin(arg1) + c[21] * sin(arg2) - c[22] * pow(rply, c[10]));

	// Declarations 

	double nsft;
	double KroneckerDelta;
	int odd;
	int oddminus;
	int oddplus;
	int even;
	int evenminus;
	int evenplus;
	int eveni;
	int evenj;
	int oddi;
	int oddj;
	double npo;
	double nmo;
	double npowmo;
	double tnpowpo;
	double npth;
	double nmt;
	double npt;
	double tnpo;
	double tnmo;
	double tnpt;
	double osubn;
	double osubntm;
	double osubntms;
	int Indexnij;
	int Indexijn;
	int Indexjin;

	// DividerForCoupling terms

	double DividerModes = -rply;

	double DividerTransMotion = 0.0; // 1.0 / (y[0] - 9.0 / 5.0 * y[2]);

	double SubDividerVol = 0.0;
	double SubDividerVolTemp = 0.0;

	double DividerVol = 0.0;


	for (int n = 0; n < NumberOfModes; n++)
	{
		nsft = ((double)n + 2.0);
		even = (2 * n + 2);

		SubDividerVolTemp = (nsft - 3.0) / ((2.0 * nsft + 1.0) * (nsft + 1.0)) * y[even] * y[even] * rply;
		SubDividerVol += SubDividerVolTemp;
	}

	DividerVol = 1.0 / (SubDividerVol - D);


	//Linear coupling terms according to Shaw

	int TransSpeed = (2 * NumberOfModes + 3);

	double LinearCoupleTransMotion = 0.0; // DividerTransMotion* (-18.0 * c[14] * y[TransSpeed] * rply - 3.0 * y[1] * y[TransSpeed]);

	double LinearCouplesModesFirst[NumberOfModes];

	for (int n = 0; n < NumberOfModes; n++)
	{
		// Parameter handling

		nsft = ((double)n + 2.0);
		even = (2 * n + 2);
		odd = (2 * n + 3);

		// Constans:

		nmo = (nsft - 1.0);
		npo = (nsft + 1.0);
		npowmo = (nsft * nsft - 1.0);
		npth = (nsft + 2.0);
		tnpo = (2.0 * nsft + 1.0);

		LinearCouplesModesFirst[n] = (3.0 * y[1] * y[odd] + npowmo * npth * c[18] * rply * rply * y[even] + 2.0 * c[14] * npth * rply * (nmo * y[1] * rply * y[even] + tnpo * y[odd]));
	}


	// Explicit coupling terms :

	double VolFrt = 0.0;
	double VolFrtTemp = 0.0;
	double VolSc = 0.0;
	double VolScTemp = 0.0;

	double TmCoupModFrt[NumberOfModes];
	double TmCoupModSc[NumberOfModes];
	double TmSc = 0.0;
	double TmScTemp = 0.0;
	double TmTh = c[14] * rply * rply * (36.0 * y[TransSpeed] * y[2]);
	double TmFo = 0.0;
	double TmFoTemp = 0.0;

	double ModeFrt[NumberOfModes];
	double ModeFrtTemp[NumberOfModes];
	double ModeSc[NumberOfModes];
	double ModeScTemp[NumberOfModes];

	for (int i = 0; i < NumberOfModes; i++)
	{
		ModeFrt[i] = 0.0;
		ModeFrtTemp[i] = 0.0;
		ModeSc[i] = 0.0;
		ModeScTemp[i] = 0.0;
	}


	// Coupled terms for Translation Motion

	double TmFrt = 0.0; // y[TransSpeed] * (9.0 / 5.0 * y[3] + 18.0 / 5.0 * y[2] * y[1] * rply);

	double nsubf = 9.0 / 4.0;


	for (int n = 0; n < NumberOfModes; n++)
	{
		// Shifted parameters :
		nsft = ((double)n + 2);

		// Parameters handling :
		KroneckerDelta = 0.0;
		odd = (2 * n + 3);
		oddminus = (2 * n + 1);
		oddplus = (2 * n + 5);

		even = (2 * n + 2);
		evenminus = (2 * n);
		evenplus = (2 * n + 4);

		if (n == 0)
		{
			KroneckerDelta = 1.0;
			evenminus = 2;
			oddminus = 3;
		}

		if (n == (NumberOfModes - 1))
		{
			evenplus = 10;
			oddplus = 11;
		}

		// Constans :
		npo = (nsft + 1.0);
		nmo = (nsft - 1.0);
		npowmo = (nsft * nsft - 1.0);
		tnpowpo = (2.0 * nsft * nsft + 1.0);
		npth = (nsft + 2.0);
		nmt = (nsft - 3.0);
		npt = (nsft + 3.0);
		tnpo = (2.0 * nsft + 1.0);
		tnmo = (2.0 * nsft - 1.0);
		tnpt = (2.0 * nsft + 3.0);
		osubn = (1.0 / ((2.0 * nsft + 1.0) * (nsft + 1.0)));
		osubntm = (1.0 / ((2.0 * nsft + 1.0) * (2.0 * nsft + 3.0)));
		osubntms = (1.0 / (tnpo * tnmo));

		// Coupled terms for volume osc.

		VolFrtTemp = osubn * ((nsft + 1.5) * y[odd] * y[odd] + y[even] * rply * (2.0 * c[14] * rply * ((nsft * nsft + 5.0 * nsft + 2.0) * y[odd] - 4.0 * nsft * nsft * y[1] * rply * y[even]) \
			- nmt * (2.0 * y[1] * y[odd] + 0.5 * y[1] * y[1] * rply * y[even])));
		VolFrt += VolFrtTemp;

		VolScTemp = y[even] * y[even] / tnpo;
		VolSc += VolScTemp;


		// Coupled terms for modes

		for (int i = 0; i < NumberOfModes; i++)
		{
			for (int j = 0; j < NumberOfModes; j++)
			{
				eveni = (2 * i + 2);
				evenj = (2 * j + 2);
				oddi = (2 * i + 3);
				oddj = (2 * j + 3);
				Indexnij = (j + NumberOfModes * i + NumberOfModes * NumberOfModes * n);
				Indexjin = (n + NumberOfModes * i + NumberOfModes * NumberOfModes * j);

				ModeFrtTemp[n] = 0.5 * tnpo * npo * (y[eveni] * y[evenj] * rply * rply * (0.5 * y[1] * y[1] * ma[Indexnij] + ia[Indexnij] * c[15] * PressureTermMod) + 0.5 * (y[1] * rply * y[oddj] * y[eveni] * mb[Indexnij] + y[oddi] * y[oddj] * md[Indexnij]));
				ModeFrt[n] += ModeFrtTemp[n];

				ModeScTemp[n] = -c[14] * npo * rply * rply * (0.25 * tnpo * (y[oddj] * y[eveni] * (qb[Indexnij] + qb[Indexjin]) + y[1] * rply * y[eveni] * y[evenj] * qc[Indexjin]));
				ModeSc[n] += ModeScTemp[n];

			}
		}

		// TransMotion Coupling for Modes

		TmCoupModFrt[n] = 0.0; // (nsubf * KroneckerDelta * y[TransSpeed] * y[TransSpeed] - 3.0 * npo * y[TransSpeed] * (tnpo / (2.0 * tnpt) * y[oddplus] + (1.0 - KroneckerDelta) * (y[1] * rply * y[evenminus] + 0.5 * y[oddminus])));

		TmCoupModSc[n] = 0.0; // (-c[14] * rply * rply * npo * (y[TransSpeed] * 3.0 * nsft * (2.0 * npo * npth / tnpt * y[evenplus] - (1.0 - KroneckerDelta) * tnpowpo / tnmo * y[evenminus])));

		// Coupled terms for Translation Motion

		TmScTemp = 0.0; //-(9.0 * osubntm * (2.0 * y[1] * rply * (y[oddplus] * y[even] - 2.0 * nsft * y[odd] * y[evenplus]) - 2.0 * nsft * y[even] * y[evenplus] * y[1] * y[1] * rply * rply + y[oddplus] * y[odd]));
		TmSc += 0.0; // TmScTemp;

		TmFoTemp = 0.0; // 18.0 * nsft * npo * osubntm * (9.0 * y[1] * rply * y[even] * y[evenplus] + 2.0 * npth * y[odd] * y[evenplus]) - (18.0 * nsft * tnpowpo) * osubntms * y[odd] * y[evenminus];
		TmFo += 0.0; // TmFoTemp;
	}

	double ExplicitVol = -N - 0.25 * y[TransSpeed] * y[TransSpeed] + VolFrt + PressureTermVol * c[15] * rply * rply * VolSc;

	TmFo = 0.0; // c[14] * rply* rply* TmFo;

	// Iteration for inplicit parts :

	double jsft;
	double jmo;
	double jpowmo;
	double jpth;
	double tjpo;

	double InplicitVol;
	double InplicitVolTemp;

	double LinearCouplModSc[NumberOfModes];
	double TmCoupModTh[NumberOfModes];
	double TmInplicit;
	double TmInplicitTemp;
	double SubInplicitMod[NumberOfModes];
	double InplicitMod[NumberOfModes];
	double TempSubImplicit[NumberOfModes];
	double TempSubImplicitTemp;
	double LinearCouplesModesWithJ;
	double fn[SysDim];
	double IterError = 300.0;
	double IterErrorTemp;
	int NCounter = 0;

	while (IterError >= AbsTolerance)
	{
		InplicitVol = 0.0;
		InplicitVolTemp = 0.0;
		TmInplicit = 0.0;

		for (int h = 0; h < NumberOfModes; h++)
		{
			TempSubImplicit[h] = 0.0;
		}

		for (int n = 0; n < NumberOfModes; n++)
		{
			// Shifted parameters :
			nsft = ((double)n + 2);

			// Parameters for the arrays :
			odd = (2 * n + 3);
			even = (2 * n + 2);

			// Constans :

			osubn = 1.0 / ((2.0 * nsft + 1.0) * (nsft + 1.0));
			nmo = (nsft - 1.0);
			npt = (nsft + 3.0);
			npth = (nsft + 2.0);
			tnpo = (2.0 * nsft + 1.0);

			InplicitVolTemp = osubn * npt * y[even] * DividerModes * (LinearCouplesModesFirst[n] - ((nsft - 1) * f[1] * y[even]));
			InplicitVol += InplicitVolTemp;
		}

		fn[1] = DividerVol * (InplicitVol + ExplicitVol);

		for (int n = 0; n < NumberOfModes; n++)
		{
			KroneckerDelta = 0.0;
			nsft = ((double)n + 2);

			odd = (2 * n + 3);
			even = (2 * n + 2);
			evenminus = (2 * n);
			evenplus = (2 * n + 4);
			oddplus = (2 * n + 5);

			if (n == 0)
			{
				KroneckerDelta = 1.0;
				evenminus = 2;
			}

			if (n == (NumberOfModes - 1))
			{
				evenplus = 10;
				oddplus = 11;
			}

			// Constans :

			npo = (nsft + 1.0);
			nmo = (nsft - 1.0);
			tnpo = (2.0 * nsft + 1.0);
			tnmo = (2.0 * nsft - 1.0);
			tnpt = (2.0 * nsft + 3.0);

			LinearCouplModSc[n] = -nmo * f[1] * y[even];
			TmCoupModTh[n] = 0.0; // npo* (1.5 * nsft * LinearCoupleTransMotion * ((1.0 - KroneckerDelta) * y[evenminus] / tnmo - y[evenplus] / tnpt));

			for (int i = 0; i < NumberOfModes; i++)
			{
				for (int j = 0; j < NumberOfModes; j++)
				{
					jsft = ((double)j + 2);
					oddi = (2 * i + 3);
					eveni = (2 * i + 2);
					oddj = (2 * j + 3);
					evenj = (2 * j + 2);

					jmo = (jsft - 1.0);
					jpowmo = (jsft * jsft - 1.0);
					jpth = (jsft + 2.0);
					tjpo = (2.0 * jsft + 1.0);

					Indexnij = j + NumberOfModes * i + NumberOfModes * NumberOfModes * n;
					Indexijn = n + NumberOfModes * j + NumberOfModes * NumberOfModes * i;

					LinearCouplesModesWithJ = DividerModes * (3.0 * y[1] * y[oddj] + jpowmo * jpth * c[18] * rply * rply * y[evenj] + 2.0 * c[14] * jpth * rply * (jmo * y[1] * rply * y[evenj] + tjpo * y[oddj]) - jmo * f[1] * y[evenj]);

					TempSubImplicitTemp = (f[1] * rply * y[eveni] * y[evenj] * gd[Indexijn] + y[eveni] * mc[Indexnij] * LinearCouplesModesWithJ);
					TempSubImplicit[n] += TempSubImplicitTemp;
				}
			}

			SubInplicitMod[n] = npo * 0.25 * tnpo * TempSubImplicit[n];
			InplicitMod[n] = LinearCouplModSc[n] + TmCoupModTh[n] + SubInplicitMod[n];

			fn[odd] = DividerModes * (LinearCouplesModesFirst[n] + TmCoupModFrt[n] + ModeFrt[n] + TmCoupModSc[n] + ModeSc[n] + InplicitMod[n]);

			TmInplicitTemp = 0.0; // -9.0 * osubntm * (-2.0 * nsft * y[even] * y[evenplus] * f[1] * rply + npo * f[oddplus] * y[even] - nsft * f[odd] * y[evenplus]);
			TmInplicit += 0.0; // TmInplicitTemp;
		}
		fn[3 + 2 * NumberOfModes] = 0.0; // LinearCoupleTransMotion + DividerTransMotion * (TmFrt + TmSc + TmTh + TmFo + TmInplicit);
		//f[1] = DividerVol * (InplicitVol + ExplicitVol);

		IterError = 0.0;
		for (int i = 0; i < SysDim / 2; i++)
		{
			IterErrorTemp = abs(f[2 * i + 1] - fn[2 * i + 1]);
			if (IterErrorTemp > IterError)
			{
				IterError = IterErrorTemp;
			}
			//std::cout << IterErrorTemp[i] << std::endl;
		}
		//std::cout << "ErrorMax "<< ErrorInitial << std::endl;

		for (int i = 0; i < SysDim / 2; i++)
		{
			f[2 * i + 1] = fn[2 * i + 1];
		}

		NCounter += 1;

		if (NCounter == 250)
		{
			//std::cout << "Fixed-point iteration did not converge" << std::endl;
			break;
		}
	}
}

// Runge-Kutta-Cash-Karp Method for K-M

void rkckKM(double& t0, double* y, double* c, double& dt, double* yn, double* error)
{

	double k1[SysDimKM];
	double k2[SysDimKM];
	double k3[SysDimKM];
	double k4[SysDimKM];
	double k5[SysDimKM];
	double k6[SysDimKM];
	double yact[SysDimKM];


	//k1

	OdeFunKM(k1, dt, t0, y, c);

	for (int i = 0; i < SysDimKM; i++)
	{
		k1[i] = dt * k1[i];
	}

	//k2

	double hk2;
	hk2 = t0 + (1.0 / 5.0) * dt;

	for (int i = 0; i < SysDimKM; ++i)
	{
		yact[i] = y[i] + (1.0 / 5.0) * k1[i];
	}

	OdeFunKM(k2, dt, hk2, yact, c);

	for (int i = 0; i < SysDimKM; i++)
	{
		k2[i] = dt * k2[i];
	}

	//k3

	double hk3;
	hk3 = t0 + (3.0 / 10.0) * dt;

	for (int i = 0; i < SysDimKM; ++i)
	{
		yact[i] = y[i] + (3.0 / 40.0) * k1[i] + (9.0 / 40.0) * k2[i];
	}

	OdeFunKM(k3, dt, hk3, yact, c);

	for (int i = 0; i < SysDimKM; i++)
	{
		k3[i] = dt * k3[i];
	}

	//k4

	double hk4;
	hk4 = t0 + (3.0 / 5.0) * dt;

	for (int i = 0; i < SysDimKM; ++i)
	{
		yact[i] = y[i] + (3.0 / 10.0) * k1[i] + (-9.0 / 10.0) * k2[i] + (6.0 / 5.0) * k3[i];
	}

	OdeFunKM(k4, dt, hk4, yact, c);

	for (int i = 0; i < SysDimKM; i++)
	{
		k4[i] = dt * k4[i];
	}

	//k5

	double hk5;
	hk5 = t0 + (1.0) * dt;

	for (int i = 0; i < SysDimKM; ++i)
	{
		yact[i] = y[i] + (-11.0 / 54.0) * k1[i] + (5.0 / 2.0) * k2[i] + (-70.0 / 27.0) * k3[i] + (35.0 / 27.0) * k4[i];
	}

	OdeFunKM(k5, dt, hk5, yact, c);

	for (int i = 0; i < SysDimKM; i++)
	{
		k5[i] = dt * k5[i];
	}

	//k6

	double hk6;
	hk6 = t0 + (7.0 / 8.0) * dt;

	for (int i = 0; i < SysDimKM; ++i)
	{
		yact[i] = y[i] + (1631.0 / 55296.0) * k1[i] + (175.0 / 512.0) * k2[i] + (575.0 / 13824.0) * k3[i] + (44275.0 / 110592.0) * k4[i] + (253.0 / 4096.0) * k5[i];
	}

	OdeFunKM(k6, dt, hk6, yact, c);

	for (int i = 0; i < SysDimKM; i++)
	{
		k6[i] = dt * k6[i];
	}

	// yn

	for (int i = 0; i < SysDimKM; ++i)
	{
		yn[i] = y[i] + (37.0 / 378.0) * k1[i] + (0.0) * k2[i] + (250.0 / 621.0) * k3[i] + (125.0 / 594.0) * k4[i] + (0.0) * k5[i] + (512.0 / 1771.0) * k6[i];
	}

	// error

	for (int i = 0; i < SysDimKM; ++i)
	{
		error[i] = abs(((37.0 / 378.0) - (2825.0 / 27648.0)) * k1[i] + (0.0) * k2[i] + ((250.0 / 621.0) - (18575.0 / 48384.0)) * k3[i] + ((125.0 / 594.0) - (13525.0 / 55296.0)) * k4[i] + ((0.0) - (277.0 / 14336.0)) * k5[i] + ((512.0 / 1771.0) - (1.0 / 4.0)) * k6[i]);

		if (error[i] == 0.0)
		{
			error[i] = 1.0e-20;
		}
	}
}

// Runge-Kutta-Cash-Karp Method for K-M and the coupling terms

void rkck(double& t0, double* y, double* c, double* rp, double& dt, double* yn, double* error, double* ia, double* gd, double* ma, double* mb, double* mc, double* md, double* qb, double* qc)
{

	double k1[SysDim];
	double k2[SysDim];
	double k3[SysDim];
	double k4[SysDim];
	double k5[SysDim];
	double k6[SysDim];
	double yact[SysDim];


	//k1

	OdeFun(k1, dt, t0, y, c, rp, ia, gd, ma, mb, mc, md, qb, qc);

	for (int i = 0; i < SysDim; i++)
	{
		k1[i] = dt * k1[i];
	}

	//k2

	double hk2;
	hk2 = t0 + (1.0 / 5.0) * dt;

	for (int i = 0; i < SysDim; ++i)
	{
		yact[i] = y[i] + (1.0 / 5.0) * k1[i];
	}

	OdeFun(k2, dt, hk2, yact, c, rp, ia, gd, ma, mb, mc, md, qb, qc);

	for (int i = 0; i < SysDim; i++)
	{
		k2[i] = dt * k2[i];
	}

	//k3

	double hk3;
	hk3 = t0 + (3.0 / 10.0) * dt;

	for (int i = 0; i < SysDim; ++i)
	{
		yact[i] = y[i] + (3.0 / 40.0) * k1[i] + (9.0 / 40.0) * k2[i];
	}

	OdeFun(k3, dt, hk3, yact, c, rp, ia, gd, ma, mb, mc, md, qb, qc);

	for (int i = 0; i < SysDim; i++)
	{
		k3[i] = dt * k3[i];
	}

	//k4

	double hk4;
	hk4 = t0 + (3.0 / 5.0) * dt;

	for (int i = 0; i < SysDim; ++i)
	{
		yact[i] = y[i] + (3.0 / 10.0) * k1[i] + (-9.0 / 10.0) * k2[i] + (6.0 / 5.0) * k3[i];
	}

	OdeFun(k4, dt, hk4, yact, c, rp, ia, gd, ma, mb, mc, md, qb, qc);

	for (int i = 0; i < SysDim; i++)
	{
		k4[i] = dt * k4[i];
	}

	//k5

	double hk5;
	hk5 = t0 + (1.0) * dt;

	for (int i = 0; i < SysDim; ++i)
	{
		yact[i] = y[i] + (-11.0 / 54.0) * k1[i] + (5.0 / 2.0) * k2[i] + (-70.0 / 27.0) * k3[i] + (35.0 / 27.0) * k4[i];
	}

	OdeFun(k5, dt, hk5, yact, c, rp, ia, gd, ma, mb, mc, md, qb, qc);

	for (int i = 0; i < SysDim; i++)
	{
		k5[i] = dt * k5[i];
	}

	//k6

	double hk6;
	hk6 = t0 + (7.0 / 8.0) * dt;

	for (int i = 0; i < SysDim; ++i)
	{
		yact[i] = y[i] + (1631.0 / 55296.0) * k1[i] + (175.0 / 512.0) * k2[i] + (575.0 / 13824.0) * k3[i] + (44275.0 / 110592.0) * k4[i] + (253.0 / 4096.0) * k5[i];
	}

	OdeFun(k6, dt, hk6, yact, c, rp, ia, gd, ma, mb, mc, md, qb, qc);

	for (int i = 0; i < SysDim; i++)
	{
		k6[i] = dt * k6[i];
	}

	// yn

	for (int i = 0; i < SysDim; ++i)
	{
		yn[i] = y[i] + (37.0 / 378.0) * k1[i] + (0.0) * k2[i] + (250.0 / 621.0) * k3[i] + (125.0 / 594.0) * k4[i] + (0.0) * k5[i] + (512.0 / 1771.0) * k6[i];
	}

	// error

	for (int i = 0; i < SysDim; ++i)
	{
		error[i] = abs(((37.0 / 378.0) - (2825.0 / 27648.0)) * k1[i] + (0.0) * k2[i] + ((250.0 / 621.0) - (18575.0 / 48384.0)) * k3[i] + ((125.0 / 594.0) - (13525.0 / 55296.0)) * k4[i] + ((0.0) - (277.0 / 14336.0)) * k5[i] + ((512.0 / 1771.0) - (1.0 / 4.0)) * k6[i]);

		if (error[i] == 0.0)
		{
			error[i] = 1.0e-20;
		}
	}
}

//Define tolerances

template<int SD>
void GetTolerance(double* y, double* yn, double* tol)
{

	double abstol[SD];
	double reltol[SD];
	double reltolacty[SD];
	double reltolactyn[SD];
	double reltolact[SD];


	for (int i = 0; i < SD; i++)
	{
		abstol[i] = AbsTolerance;
		reltol[i] = RelTolerance;
	}

	for (int x = 0; x < SD; ++x)
	{
		reltolacty[x] = reltol[x] * abs(y[x]);
		reltolactyn[x] = reltol[x] * abs(yn[x]);
	}

	for (int x = 0; x < SD; x++)
	{
		reltolact[x] = std::min(reltolacty[x], reltolactyn[x]);
	}


	for (int i = 0; i < SD; i++)
	{
		if (abstol[i] >= reltolact[i])
		{
			tol[i] = abstol[i];
		}

		else
		{
			tol[i] = reltolact[i];
		}
	}
}

//Calculate the following time step

template<int SD>
void GetTimeStep(double* tol, double* error, double* y, double* yn, double& t0, double& dt, double& t, double* Rm, double* Rt)
{
	double MintolDivError = 1.0e300;
	double TimeStepper;
	double MaxTimeStep = 1.0e6;
	double MinTimeStep = 1.0e-12;
	bool Update = false;

	
	for (int i = 0; i < SD; ++i)
	{
		if ((tol[i] / error[i]) < MintolDivError)
		{
			MintolDivError = (tol[i] / error[i]);
		}
	}

	if (MintolDivError >= 1)
	{
		Update = 1;
	}

	if (Update == 1)
	{
		TimeStepper = 0.9 * pow(MintolDivError, 0.2);
	}

	else
	{
		TimeStepper = 0.9 * pow(MintolDivError, 0.25);
	}

	if (isfinite(TimeStepper) == 0)
	{
		Update = 0;
	}

	if (Update == 1)
	{
		for (int x = 0; x < SD; ++x)
		{
			y[x] = yn[x];
		}

		if (SD == SysDim)
		{
			for (int i = 0; i < NumberOfModes; i++)
			{
				double RSubA = y[2 * i + 2] / y[0];
				if (RSubA > 0.9)
				{
				    AvoidCrash = 1;
					std::cout<<"R/A > 0.9, simulation aborted"<<std::endl;
					return;
				}
			}
		}

		t0 += dt;
		t += dt;

	}

	TimeStepper = std::min(TimeStepper, GrowLimit);
	TimeStepper = std::max(TimeStepper, ShrinkLimit);

	dt = dt * TimeStepper;

	dt = std::min(dt, MaxTimeStep);
	dt = std::max(dt, MinTimeStep);
	

	if (dt <= MinTimeStep)
	{
		std::cout << "Minimum Time Step is under 1e-12, simulation aborted" << std::endl;
		AvoidCrash = 1;
		return;
	}

	if ((t + dt) > T)
	{
		dt = T - t;
	}
	
	for (int i = 0; i < SD; i++)
	{
		if (abs(y[i]/y[0]) > abs(Rm[i]))
		{
			Rm[i] = y[i]/y[0];
			
		}
	}

	for (int i = 0; i < SD; i++)
	{
		if (abs(y[i]) > abs(Rt[i]))
		{
			Rt[i] = y[i];
		}
	}

}

// ODE Solver

void OdeSolver(double& t0, double* y, double* c, double* rp, double* Rm, double* Rt, double* ia, double* gd, double* ma, double* mb, double* mc, double* md, double* qb, double* qc)
{

	double t = 0.0;
	double dt = 3.016e-6;
	double yn[SysDim];
	double error[SysDim];
	double tol[SysDim];

	for (int i = 0; i < SysDim; i++)
	{
		Rm[i] = 0.0;
	}

	while (t < T)
	{
		rkck(t0, y, c, rp, dt, yn, error, ia, gd, ma, mb, mc, md, qb, qc);
		GetTolerance<SysDim>(y, yn, tol);
		GetTimeStep<SysDim>(tol, error, y, yn, t0, dt, t, Rm, Rt);
		if (AvoidCrash == 1)
		{
			return;
		}
	}
}

void OdeSolverKM(double& t0, double* y, double* c, double* rp, double* Rm, double* Rt)
{
	double t = 0.0;
	double dt = 3.016e-6;
	double yn[SysDimKM];
	double error[SysDimKM];
	double tol[SysDimKM];


	for (int i = 0; i < SysDimKM; i++)
	{
		Rm[i] = 0.0;
		Rt[i] = 0.0;
	}

	while (t < T)
	{
		rkckKM(t0, y, c, dt, yn, error);
		GetTolerance<SysDimKM>(y, yn, tol);
		GetTimeStep<SysDimKM>(tol, error, y, yn, t0, dt, t, Rm, Rt);
		if (AvoidCrash == 1)
		{
			return;
		}
	}
}
