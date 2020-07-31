#ifndef SOURCE    // To make sure you don't declare the function more than once by including the header multiple times.
#define SOURCE
#include "stdafx.h"

struct Graph {
	int V;
	struct AdjList* array;
};
int* PrimMST(struct Graph* graph);
struct Graph* createGraph(int V);
void addEdge(struct Graph* graph, int src, int dest, int weight);
#endif