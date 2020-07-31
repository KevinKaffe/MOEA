
#include "stdafx.h"
#include "Source.h"
#include <vector>
#include <math.h> 
#include <algorithm>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include "cifar/cifar10_reader.hpp"
#include <thread> 
#include <chrono>
#include <iostream>
#include <fstream>
using namespace std;
using namespace cv;

const double MR = 0.2;
const double CP = 0.7;
struct Segment {

};
enum direction{
	LEFT=0,
	RIGHT,
	UP,
	DOWN,
	NONE
};

struct Vertex {
	int distance;
	int parent;
	int child;
	bool visited = false;
	bool selected = false;
};

struct Node {
	int id = 0;
	int segment=0;
	Node * neighbours[4];
	bool visited = false;
};

struct Chromosome {
	vector<direction> directions;
	int width;
	int height;
	double pixelDistance;
	double connectivity;
	double deviation;
	double fitness;
	vector<int> border;
	int paretoRank=1;
	double c_dist = 0;
	int largestSegment;
};

struct chrom_sort {
	bool operator()(const Chromosome &left, const Chromosome &right) {
		return left.fitness > right.fitness;
	}
};

struct conn_sort {
	bool operator()(const Chromosome &left, const Chromosome &right) {
		return left.connectivity < right.connectivity;
	}
};

struct pix_sort {
	bool operator()(const Chromosome &left, const Chromosome &right) {
		return left.pixelDistance > right.pixelDistance;
	}
};

struct p_conn_sort {
	bool operator()(const Chromosome *left, const Chromosome *right) {
		return left->connectivity < right->connectivity;
	}
};

struct p_pix_sort {
	bool operator()(const Chromosome *left, const Chromosome *right) {
		return left->pixelDistance > right->pixelDistance;
	}
};

struct dev_sort {
	bool operator()(const Chromosome &left, const Chromosome &right) {
		return left.deviation > right.deviation;
	}
};

struct crowd_sort {
	bool operator()(const Chromosome &left, const Chromosome &right) {
		return left.c_dist < right.c_dist;
	}
};
int neighbours(Node* node, int segment) {
	node -> visited = true;
	node -> segment = segment;
	int sum = 1;
	for (Node* neighbour : node->neighbours) {
		if (neighbour != NULL && neighbour -> visited == false) {
			sum+=neighbours(neighbour, segment);
		}	
	}
	return sum;
}
int segmentation(vector<Node*> &nodes) {
	int segment = 0;
	map<int, int> segmentSize;
	for (Node* node : nodes) {
		if (node -> visited == false) {
			segmentSize[segment]= neighbours(node, segment);
			segment++;
		}
	}
	return --segment;
}

vector<Node*> graphing(Chromosome* c) {
	vector<Node*> nodes;
	for (unsigned int i = 0; i < c->directions.size(); i++)
		nodes.push_back(new Node());
	for (int y = 0; y < c->height; y++) {
		for (int x = 0; x < c->width; x++) {
			int i = x + y*c->width;
			Node* n = nodes[i];
			int dir = c->directions[i];
			switch (dir) {
			case LEFT:
				if (x != 0) {
					Node* neighbour = nodes[x - 1 + y*c->width];
					n->neighbours[0] = neighbour;
					neighbour->neighbours[1] = n;
				}
				break;
			case RIGHT:
				if (x != c->width - 1) {
					Node* neighbour = nodes[x + 1 + y*c->width];
					n->neighbours[1] = neighbour;
					neighbour->neighbours[0] = n;
				}
				break;
			case UP:
				if (y != 0) {
					Node* neighbour = nodes[x + (y - 1)*c->width];
					n->neighbours[2] = neighbour;
					neighbour->neighbours[3] = n;
				}
				break;
			case DOWN:
				if (y != c->height - 1) {
					Node* neighbour = nodes[x + (y + 1)*c->width];
					n->neighbours[3] = neighbour;
					neighbour->neighbours[2] = n;
				}
				break;
			case NONE:
				break;
			}

		}
	}

	return nodes;
}



double rgbDistance(int r1, int r2, int g1, int g2, int b1, int b2) {
	return sqrt(pow(r1 - r2,2) + pow(g1 - g2,2) + pow(b1 - b2,2));
}

void imageToGraph(struct Graph* graph, Mat image) {
	int channels = image.channels();
	int nRows = image.rows;
	int nCols = image.cols;
	uchar* old = image.ptr<uchar>(0);
	for (int i = 0; i < nRows; i++) {
		uchar* p = image.ptr<uchar>(i);
		for (int j = 0; j < nCols; j++) {
			if (i > 0) {
				addEdge(graph, i*nCols + j, (i - 1)*nCols + j, rgbDistance(p[3 * j], old[3 * j], p[3 * j + 1], old[3 * j + 1], p[3 * j + 2], old[3 * j + 2]));
			}
			if (j > 0) {
				addEdge(graph, i*nCols + j, i*nCols + j-1, rgbDistance(p[3 * j], p[3 * (j-1)], p[3 * j + 1], p[3 * (j - 1) +1], p[3 * j + 2], p[3 * (j - 1) +2]));
			}
		}
		old = p;
	}
}

Chromosome mstToChromosome(int* mst, int V) {
	Chromosome c;
	for (int i = 0; i < V; i++) {
		if (mst[i] == -1) {
			c.directions.push_back(NONE);
		}
		else if (mst[i] == i-1) {
			c.directions.push_back(LEFT);
		}
		else if (mst[i] == i+1) {
			c.directions.push_back(RIGHT);
		}
		else if (mst[i] < i) {
			c.directions.push_back(UP);
		}
		else if (mst[i] > i) {
			c.directions.push_back(DOWN);
		}
		else {
			cout << "ERR" << endl;
		}
	}
	return c;
}

double pixelDistance(vector<Node*> nodes, Mat* image, vector<int> border) {
	double sum = 0;
	int nCols = image->cols;
	int nRows = image->rows;
	uchar* p = image->ptr<uchar>(0);
	uchar* p0 = image->ptr<uchar>(0);
	uchar* p1 = image->ptr<uchar>(0);
	int prevY = 0;
	for (int i : border) {
		int x = i%nCols;
		int y = i / nCols;
		if (y != prevY) {
			p = image->ptr<uchar>(y);
			p0 = image->ptr<uchar>(y - 1);
			if (y != nRows - 1) {
				p1 = image->ptr<uchar>(y + 1);
			}
			prevY = y;
		}
		if (x != 0 && nodes[x - 1 + y*nCols]->segment != nodes[x + y*nCols]->segment) {
			sum += rgbDistance(p[3 * x], p[3 * (x - 1)], p[3 * x + 1], p[3 * (x - 1) + 1], p[3 * x + 2], p[3 * (x - 1) + 2]);
			if (y != 0 && nodes[x - 1 + (y - 1)*nCols]->segment != nodes[x + y*nCols]->segment) {
				sum += rgbDistance(p[3 * x], p0[3 * (x - 1)], p[3 * x + 1], p0[3 * (x - 1) + 1], p[3 * x + 2], p0[3 * (x - 1) + 2]);
			}
			if (y != nRows - 1 && nodes[x - 1 + (y + 1)*nCols]->segment != nodes[x + y*nCols]->segment) {
				sum += rgbDistance(p[3 * x], p1[3 * (x - 1)], p[3 * x + 1], p1[3 * (x - 1) + 1], p[3 * x + 2], p1[3 * (x - 1) + 2]);
			}
		}
		if (x != nCols - 1 && nodes[x + 1 + y*nCols]->segment != nodes[x + y*nCols]->segment) {
			sum += rgbDistance(p[3 * x], p[3 * (x + 1)], p[3 * x + 1], p[3 * (x + 1) + 1], p[3 * x + 2], p[3 * (x + 1) + 2]);
			if (y != 0 && nodes[x + 1 + (y - 1)*nCols]->segment != nodes[x + y*nCols]->segment) {
				sum += rgbDistance(p[3 * x], p0[3 * (x + 1)], p[3 * x + 1], p0[3 * (x + 1) + 1], p[3 * x + 2], p0[3 * (x + 1) + 2]);
			}
			if (y != nRows - 1 && nodes[x - 1 + (y + 1)*nCols]->segment != nodes[x + y*nCols]->segment) {
				sum += rgbDistance(p[3 * x], p1[3 * (x + 1)], p[3 * x + 1], p1[3 * (x + 1) + 1], p[3 * x + 2], p1[3 * (x + 1) + 2]);
			}
		}
		if (y != 0 && nodes[x + (y-1)*nCols]->segment != nodes[x + y*nCols]->segment) {
			sum += rgbDistance(p[3 * x], p0[3 * x], p[3 * x + 1], p0[3 * x + 1], p[3 * x + 2], p0[3 * x + 2]);
		}
		if (y != nRows - 1 && nodes[x  + (y+1)*nCols]->segment != nodes[x + y*nCols]->segment) {
			sum += rgbDistance(p[3 * x], p1[3 * x], p[3 * x + 1], p1[3 * x + 1], p[3 * x + 2], p1[3 * x + 2]);
		}
	}
	return sum;
}


double connectivity(const vector<Node*> &nodes, Mat* image, const vector<int> &border) {
	const double F = 0.125;
	double sum = 0;
	int nCols = image->cols;
	int nRows = image->rows;
	vector<thread> threads;

	for (int i: border) {
		int x = i%nCols;
		int y = i / nCols;
		if (x != 0) {
			if (nodes[x - 1 + y*nCols]->segment != nodes[x + y*nCols]->segment)
				sum +=  1.0;
			if (y != 0 && nodes[x-1 + (y - 1)*nCols]->segment != nodes[x + y*nCols]->segment) {
				sum += 1.0/8.0;
			}
			if (y != nRows - 1 && nodes[x-1 + (y + 1)*nCols]->segment != nodes[x + y*nCols]->segment) {
				sum +=  1.0/7.0;
			}
		}
		if (x != nCols - 1) {
			if(nodes[x + 1 + y*nCols]->segment != nodes[x + y*nCols]->segment)
				sum +=  1.0/2.0;
			if (y != 0 && nodes[x + 1 + (y - 1)*nCols]->segment != nodes[x + y*nCols]->segment) {
				sum +=1.0/6.0;
			}
			if (y != nRows - 1 && nodes[x + 1 + (y + 1)*nCols]->segment != nodes[x + y*nCols]->segment) {
				sum +=  1.0 / 5.0;
			}
		}

		if (y != 0 && nodes[x + (y-1)*nCols]->segment != nodes[x + y*nCols]->segment) {
			sum +=  1.0/4.0;
		}
		if (y != nRows - 1 && nodes[x + (y + 1)*nCols]->segment != nodes[x + y*nCols]->segment) {
			sum += 1.0 / 3.0;
		}
	}
	return sum;
}

double deviation(vector<Node*> &nodes, Mat* image, map<int, int> & segmentSize) {
	map<int, double> rAverages;
	map<int, double> gAverages;
	map<int, double> bAverages;
	int nCols = image->cols;
	
	int maxSegment = 0;
	uchar* p = image->ptr<uchar>(0);
	int prevY = 0;
	for (int y = 0; y < image->rows; y++) {
		p = image->ptr<uchar>(y);
		for (int x = 0; x < nCols; x++) {
			int i = x + y*nCols;
			int segment = nodes[i]->segment;
			int size = segmentSize[segment];
			rAverages[segment] += p[3 * x] / size;
			gAverages[segment] += p[3 * x + 1] / size;
			bAverages[segment] += p[3 * x + 2] / size;
		}
	}
	double sum = 0;
	p = image->ptr<uchar>(0);
	prevY = 0;
	for (int y = 0; y < image->rows; y++) {
		image->ptr<uchar>(y);
		for (int x = 0; x < nCols; x++) {
			int i = x + y*nCols;
			sum += rgbDistance(p[3 * x], rAverages[nodes[i]->segment], p[3 * x + 1], gAverages[nodes[i]->segment], p[3 * x + 2], bAverages[nodes[i]->segment]);
		}
	}

	return sum;
}

void mutate(Chromosome* c) {
	double r = float(rand()) / float(RAND_MAX);
	if (r < MR) {
		int gene = rand() % c->directions.size();
		direction newDir = static_cast<direction>(rand() % 5);
		c->directions[gene] = newDir;
	}
}
vector<int> border(const vector<Node*> &nodes, Mat* im, int scale=1, int rows=0, int cols = 0) {
	vector<int> border;
	int nCols = im->cols;
	int nRows = im->rows;
	if (rows != 0) {
		nRows = rows;
	}
	if (cols != 0) {
		nCols = cols;
	}
	for (int y = 0; y < nRows; y++) {
		for (int x = 0; x < nCols; x++) {
			if (x > 0 && nodes[x+y*nCols]->segment != nodes[x-1 + y*nCols]->segment) {
				border.push_back(scale*x + scale*y*nCols);
				for (int i = 0; i < scale / 2; i++) {
					border.push_back(scale*(x + (i + 1)) + scale*y*nCols);
					border.push_back(scale*(x - (i + 1)) + scale*y*nCols);
				}
			}
			else if (x < nCols - 1 && nodes[x + y*nCols]->segment != nodes[x + 1 + y*nCols]->segment) {
				border.push_back(scale*x + scale*y*nCols);
				for (int i = 0; i < scale / 2; i++) {
					border.push_back(scale*(x + (i + 1)) + scale*y*nCols);
					border.push_back(scale*(x - (i + 1)) + scale*y*nCols);
				}
			}
			else if (y >0 && nodes[x + y*nCols]->segment != nodes[x + (y-1)*nCols]->segment) {
				border.push_back(scale*x + scale*y*nCols);
				for (int i = 0; i < scale / 2; i++) {
					border.push_back(scale*x  + scale*(y + (i + 1))*nCols);
					border.push_back(scale*x + scale*(y- (i + 1))*nCols);
				}
			}
			else if (y < nRows - 1 && nodes[x + y*nCols]->segment != nodes[x + (y+1)*nCols]->segment) {
				border.push_back(scale*x + scale*y*nCols);
				for (int i = 0; i < scale / 2; i++) {
					border.push_back(scale*x + scale*(y + (i + 1))*nCols);
					border.push_back(scale*x + scale*(y - (i + 1))*nCols);
				}
			}
		}
	}
	return border;
}
void getScores(Chromosome* c, Mat* im) {
	auto start = std::chrono::system_clock::now();
	vector<Node*> nodes = graphing(c);
	const double A =1.0, B = 180, C = 0.1;
	int segmentSize = segmentation(nodes);
	c->largestSegment = segmentSize;
	vector<int> bor = border(nodes, im);
	c->border = bor;
	c->connectivity = connectivity(nodes, im, bor);
	c->deviation = 0;
	//c->deviation = deviation(nodes, im, segmentSize);
	c->pixelDistance = pixelDistance(nodes, im, bor);
	cout << "Conn " << c->connectivity << "  Dist " << c -> pixelDistance << " Deviation " << c->deviation << endl;
	c->fitness = 0.001 + A*c->pixelDistance - B*(c->connectivity) - C*c->deviation;
	for (int i = 0; i < nodes.size(); i++) {
		delete nodes[i];
	}
}

pair<Chromosome, Chromosome> crossover(Chromosome* p1, Chromosome* p2, Mat* im) {
	Chromosome c1;
	Chromosome c2;
	c1.height = p1->height;
	c2.height = p2->height;
	c1.width = p1->width;
	c2.width = p2->width;
	for (int i = 0; i < p1->directions.size(); i++) {
		double r = float(rand()) / float(RAND_MAX);
		if (r < CP) {
			c1.directions.push_back(p1->directions[i]);
			c2.directions.push_back(p2->directions[i]);
		}
		else {
			c1.directions.push_back(p2->directions[i]);
			c2.directions.push_back(p1->directions[i]);
		}
	}

	mutate(&c1);
	mutate(&c2);
	getScores(&c1, im);
	getScores(&c2, im);
	return make_pair(c1, c2);
}

vector<Chromosome> selection(vector<Chromosome> &pop, int popSize) {
	vector<Chromosome> newPopulation;
	vector<double> fitness;
	double min_fitness = -1;
	for (int i = 0; i < pop.size(); i++) {
		Chromosome* chr= &(pop[i]);
		double f = chr->fitness;
		fitness.push_back(f);
		if (min_fitness == -1 || min_fitness > f) {
			min_fitness = f;
		}
	}
	double sum = 0;
	for (int i = 0; i < fitness.size(); i++) {
		fitness[i] -= min_fitness*0.975;
		sum += fitness[i];
	}
	vector<pair<double, int>> survivors;
	for (int i = 0; i < fitness.size(); i++) {
		survivors.push_back(make_pair((double)fitness[i] / sum, i));
	}
	for (int i = 0; i < popSize; i++) {
		double r = ((double)rand() / (RAND_MAX));
		for (int j = 0; j < survivors.size(); j++) {
			r -= survivors[j].first;
			if (r < 0) {
				Chromosome c = Chromosome(pop[survivors[j].second]);
				newPopulation.push_back(c);
				break;
			}
		}
	}
	//reverse(newPopulation.population.begin(), newPopulation.population.end());

	return newPopulation;
}

void crossoverThread(Chromosome* c1, Chromosome* c2, Mat* im, int index, vector<Chromosome> *pop) {
	pair<Chromosome, Chromosome> chromosomes = crossover(c1, c2, im);
	(*pop)[2 * index] = chromosomes.first;
	(*pop)[2 * index + 1] = chromosomes.second;
}

Chromosome best(vector<Chromosome> chromosomes) {
	vector<Chromosome> cand;
	int bestScore = INT_MAX;
	for (int i = 0; i < chromosomes.size(); i++) {
		if (chromosomes[i].paretoRank <= bestScore) {
			bestScore = chromosomes[i].paretoRank;
		}
	}
	for (Chromosome c : chromosomes) {
		if (c.paretoRank == bestScore) {
			cand.push_back(c);
		}
	}
	sort(cand.begin(), cand.end(), crowd_sort());
	reverse(cand.begin(), cand.end());
	return cand[0];
}
vector<Chromosome> tournament(vector<Chromosome> population, int popSize, int tournSize) {
	vector<Chromosome> newPop;
	for (int i = 0; i < popSize; i++) {
		vector<Chromosome> cand;
		for (int j = 0; j < tournSize; j++) {
			cand.push_back(population[rand() % population.size()]);
		}
		newPop.push_back(best(cand));
	}
	return newPop;
}

bool dominated(Chromosome c, vector<Chromosome> population) {
	for (Chromosome cc : population) {

		if (cc.connectivity < c.connectivity && cc.pixelDistance > c.pixelDistance) {
			return true;
		}
	}
	return false;
}

vector<Chromosome> crowdingDistance(vector<Chromosome> chromosomes) {
	sort(chromosomes.begin(), chromosomes.end(), conn_sort());
	chromosomes[0].c_dist = INT_MAX/2;
	chromosomes[chromosomes.size() - 1].c_dist = INT_MAX/2;
	for (int i = 1; i < chromosomes.size()-1; i++) {
		chromosomes[i].c_dist += (chromosomes[i + 1].connectivity - chromosomes[i - 1].connectivity) / (chromosomes[chromosomes.size() - 1].connectivity - chromosomes[0].connectivity);
	}

	sort(chromosomes.begin(), chromosomes.end(), pix_sort());
	chromosomes[0].c_dist = INT_MAX/2;
	chromosomes[chromosomes.size() - 1].c_dist = INT_MAX/2;
	for (int i = 1; i < chromosomes.size() - 1; i++) {
		chromosomes[i].c_dist += (chromosomes[i + 1].pixelDistance - chromosomes[i - 1].pixelDistance) / (chromosomes[chromosomes.size() - 1].pixelDistance - chromosomes[0].pixelDistance);
	}


	return chromosomes;

}
vector<Chromosome> paretoSelect(vector<Chromosome> population, int popSize) {
	vector<Chromosome> newPop;
	int rank = 1;
	int n = population.size();
	map<int, vector<Chromosome>> c_map;
	for (int i = 0; i < population.size(); i++) {
		population[i].paretoRank = 0;
	}
	while (n > 0) {
		for (int i = 0; i < population.size(); i++) {
			if (!dominated(population[i], population)) {
				population[i].paretoRank = rank;
				if (population[i].largestSegment <= 1 || population[i].largestSegment>=55)
					population[i].paretoRank++;
			}
		}
		int rem = 0;
		vector<int> ids;
		for (int i = 0; i < population.size(); i++) {
			if (population[i].paretoRank !=0) {
				n--;
				c_map[rank].push_back(population[i]);
				ids.push_back(i);

			}
		}
		reverse(ids.begin(), ids.end());
		for (int i : ids) {
			population[i] = population.back();
			population.pop_back();
		}
		rank++;
	}
	rank--;
	for (int i = 1; i <= rank; i++) {
		for (Chromosome c : crowdingDistance(c_map[i]))
			newPop.push_back(c);
	}
	sort(newPop.begin(), newPop.end(), crowd_sort());
	cout << "Sorted" << endl;
	while (newPop.size() > popSize) {
		vector<int> rems;
		for (int i = 0; i < newPop.size(); i++) {
			if (newPop[i].paretoRank == rank) {
				rems.push_back(i);
				if (newPop.size() - rems.size() <= popSize) {
					reverse(rems.begin(), rems.end());
					for (int i : rems) {
						newPop[i] = newPop.back();
						newPop.pop_back();
					}
					return newPop;
				}
			}
			}
		reverse(rems.begin(), rems.end());
		for (int i : rems) {
			newPop[i] = newPop.back();
			newPop.pop_back();
		}
		rank--;
	}
	return newPop;
}
vector<Chromosome> generation(vector<Chromosome> population, int popSize, Mat* im, bool nsga=false) {
	vector<Chromosome> newPop;
	vector<Chromosome> parents;
	vector<Chromosome> elitism;
	int elites = 3;
	if (!nsga) {
		for (int i = 0; i < 3; i++) {
			elitism.push_back(population[i]);
		}
		population = selection(population, popSize);
	}
	else {
		cout << "Tournament" << endl;
		newPop = population;
		population = tournament(population, popSize, 2);
	}
	random_shuffle(population.begin(), population.end());

	vector<thread> threads;
	cout << "Crossovers" << endl;
	for (int i = 0; i < population.size()/2; i++) {
		pair<Chromosome, Chromosome> chromosomes = crossover(&population[2 * i], &population[2 * i+1], im);
		newPop.push_back(chromosomes.first);
		newPop.push_back(chromosomes.second);
	}
	if (!nsga) {
		for (int i = 0; i < elites; i++) {
			newPop.push_back(elitism[i]);
		}

	}
	else {
		random_shuffle(newPop.begin(), newPop.end());
		newPop = paretoSelect(newPop, popSize);
	}

	sort(newPop.begin(), newPop.end(), chrom_sort());
	while (newPop.size() > popSize) {
		newPop.pop_back();
	}
	cout << "End gen" << endl;
	return newPop;
}

void draw(Chromosome* c, Mat orig_im, Mat raw_im, string path, bool compressed=false) {
	int nCols = orig_im.cols;
	Mat im(orig_im.size(), orig_im.type());
	orig_im.copyTo(im);
	Mat r_im(raw_im.size(),raw_im.type());
	raw_im.copyTo(r_im);


	for (int i : c->border) {
		int x = i%nCols;
		int y = i / nCols;
		uchar* p = im.ptr<uchar>(y);
		p[3*x] = 0;
		p[3*x+1] = 0;
		p[3*x+2] = 0;
	}
	imwrite(path + ".jpg", im);

	for (int y = 0; y < r_im.rows; y++) {
		uchar* p = r_im.ptr<uchar>(y);
		for (int x = 0; x < r_im.cols; x++) {
			p[3 * x] = 255;
			p[3 * x + 1] = 255;
			p[3 * x + 2] =255;
		}
	}
	vector<int> bor = c->border;
	if (compressed) {
		vector<Node*> nodes = graphing(c);
		vector<Node*> news;
		int  segmentSize = segmentation(nodes);
		for (int j = 0; j < orig_im.rows; j++) {
			for (int i = 0; i < 3 * orig_im.cols; i++) {
				Node* nn = new Node();
				Node* n = nodes[j*orig_im.cols + i / 3];
				nn->segment = n->segment;
				news.push_back(nn);

			}
			int siz = news.size();
			for (int i = 0; i < orig_im.cols; i++) {
				Node* nn = new Node();
				Node * n = nodes[j  * orig_im.cols + i];
				nn->segment = n->segment;
				news.push_back(nn);
				news.push_back(n);
				nn = new Node();
				nn->segment = n->segment;
				news.push_back(nn);
			}
			siz = news.size();
			for (int i = 0; i < 3 * orig_im.cols; i++) {
				Node* nn = new Node();
				Node* n = nodes[j*orig_im.cols + i / 3];
				nn->segment = n->segment;
				news.push_back(nn);
			}

		}
		bor = border(news, &raw_im, 1, orig_im.rows * 3, orig_im.cols * 3);
		for (int i = 0; i < news.size(); i++) {
			delete news[i];
		}

	}

	for (int i : bor) {
		int comp = 1;
		if (compressed) {
			comp = 3;
		}
		int x = (i%(comp*orig_im.cols));
		int y = (i / (comp*orig_im.cols));
		uchar* p = r_im.ptr<uchar>(y);
		//uchar* p0 = r_im.ptr<uchar>(y+1);
		//uchar* p1 = r_im.ptr<uchar>(y + 2);


		p[3 * (x)] = 0;
		p[3 * (x) + 1] = 0;
		p[3 * (x) + 2] = 0;



		/*
				p[3 * (x+1)] = 0;
		p[3 * (x+1) + 1] = 0;
		p[3 * (x+1) + 2] = 0;
		p[3 * (x + 2)] = 0;
		p[3 * (x + 2) + 1] = 0;
		p[3 * (x + 2) + 2] = 0;
		p0[3 * (x + 1)] = 0;
		p0[3 * (x + 1) + 1] = 0;
		p0[3 * (x + 1) + 2] = 0;
		p0[3 * (x)] = 0;
		p0[3 * (x) + 1] = 0;
		p0[3 * (x) + 2] = 0;
		p0[3 * (x + 2)] = 0;
		p0[3 * (x + 2) + 1] = 0;
		p0[3 * (x + 2) + 2] = 0;
		p1[3 * (x + 1)] = 0;
		p1[3 * (x + 1) + 1] = 0;
		p1[3 * (x + 1) + 2] = 0;
		p1[3 * (x)] = 0;
		p1[3 * (x)+1] = 0;
		p1[3 * (x)+2] = 0;
		p1[3 * (x + 2)] = 0;
		p1[3 * (x + 2) + 1] = 0;
		p1[3 * (x + 2) + 2] = 0;
		*/

		

	}
	imwrite(path+"_eval.jpg", r_im);
	if (compressed) {
		for (int i : c->border) {
			int x = 3 * (i % orig_im.cols);
			int y = 3 * (i / orig_im.cols);
			uchar* p = r_im.ptr<uchar>(y);
			uchar* p0 = r_im.ptr<uchar>(y + 1);
			uchar* p1 = r_im.ptr<uchar>(y + 2);


			p[3 * (x)] = 0;
			p[3 * (x)+1] = 0;
			p[3 * (x)+2] = 0;




			p[3 * (x + 1)] = 0;
			p[3 * (x + 1) + 1] = 0;
			p[3 * (x + 1) + 2] = 0;
			p[3 * (x + 2)] = 0;
			p[3 * (x + 2) + 1] = 0;
			p[3 * (x + 2) + 2] = 0;
			p0[3 * (x + 1)] = 0;
			p0[3 * (x + 1) + 1] = 0;
			p0[3 * (x + 1) + 2] = 0;
			p0[3 * (x)] = 0;
			p0[3 * (x)+1] = 0;
			p0[3 * (x)+2] = 0;
			p0[3 * (x + 2)] = 0;
			p0[3 * (x + 2) + 1] = 0;
			p0[3 * (x + 2) + 2] = 0;
			p1[3 * (x + 1)] = 0;
			p1[3 * (x + 1) + 1] = 0;
			p1[3 * (x + 1) + 2] = 0;
			p1[3 * (x)] = 0;
			p1[3 * (x)+1] = 0;
			p1[3 * (x)+2] = 0;
			p1[3 * (x + 2)] = 0;
			p1[3 * (x + 2) + 1] = 0;
			p1[3 * (x + 2) + 2] = 0;




		}
		imwrite(path + "_eval_fat.jpg", r_im);
	}



}

void initChr(struct Graph* graph, Mat im, vector<Chromosome> *population, int i) {
	im = Mat(im);
	int* mst = PrimMST(graph);
	Chromosome chr = mstToChromosome(mst, graph->V);
	chr.height = im.rows;
	chr.width = im.cols;
	getScores(&chr, &im);
	(*population)[i] = chr;
	delete[] mst;
}

Mat compress(Mat im) {
	Mat compressed(im.rows/3, im.cols/3, im.type());
	for (int y = 0; y < compressed.rows; y++) {
		uchar* new_p = compressed.ptr<uchar>(y);
		uchar* old_p = im.ptr<uchar>(3*y);
		uchar* old_p1 = im.ptr<uchar>(3 * y + 1);
		uchar* old_p2 = im.ptr<uchar>(3 * y + 2);
		for (int x = 0; x < compressed.cols; x++) {
			for (int i = 0; i < 3; i++) {
				new_p[3 * x + i] = old_p1[9 * x + 3 + i];//(old_p[3 * (3 * x) + i] + old_p[3 * (3*x + 1) + i] + old_p[3 * (3 * x + 2) + i] + old_p2[3 * 3*x +i ] + old_p2[3 * (3*x + 1) + i] + old_p2[3 * (3 * x + 2) + i] + old_p1[3 * 3 * x + i] + old_p1[3 * (3 * x + 1) + i] + old_p1[3 * (3 * x + 2) + i]) / 9;
			}
		}
	}
	return compressed;
}


int main(int argc, char** argv)
{
	bool compressed = false;
	string path = "C:/Users/Kevin/Documents/Training/text/";
	if (argc == 2) {
		path = argv[1];
	}
	Mat raw_im = imread(path+ "Test Image.png");
	Mat im;
	if (compressed) {
		im = compress(raw_im);
	}
	else {
		im = raw_im;
	}
	int popSize = 50;
	struct Graph* graph = createGraph(im.rows*im.cols);
	imageToGraph(graph, im);
	vector<Chromosome> population(popSize);
	vector<thread> threads;
	for (int i = 0; i < popSize; i++) {
		initChr(graph, im, &population, i);
		/*
		int* mst = PrimMST(graph);
		Chromosome chr = mstToChromosome(mst, graph->V);
		chr.height = im.rows;
		chr.width = im.cols;
		getScores(&chr, &im);
		population[i] = chr;
		delete[] mst;
		*/

	}
	//population = generation(population, popSize, &im);
	for (int i = 0; i <= 200; i++) {
		cout << "Generation " << i << endl;
		population = generation(population, popSize, &im, true);
		if (i % 10 == 0) {
			for (int j = 0; j < population.size(); j++) {
				draw(&population[j], im, raw_im, path + "ims/" + "Test" + to_string(i) + "_" + to_string(j), compressed);
			}
		}
		cout << "Best fitness" << population[0].fitness << " Worst: " << population[population.size()-1].fitness << endl << endl << endl;
		cout << population.size() << endl;
	}
	int x = 0;
	cin >> x;
    return 0;
}

