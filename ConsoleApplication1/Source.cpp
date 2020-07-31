#include "stdafx.h"
#include "Source.h"
#include <limits.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <iostream>
struct AdjListNode {
	int dest;
	int weight;
	struct AdjListNode* next;
};

struct AdjList {
	struct AdjListNode* head; // pointer to head node of list 
};



struct AdjListNode* newAdjListNode(int dest, int weight)
{
	struct AdjListNode* newNode = (struct AdjListNode*)malloc(sizeof(struct AdjListNode));
	newNode->dest = dest;
	newNode->weight = weight;
	newNode->next = NULL;
	return newNode;
}

struct Graph* createGraph(int V)
{
	struct Graph* graph = (struct Graph*)malloc(sizeof(struct Graph));
	graph->V = V;

	// Create an array of adjacency lists.  Size of array will be V 
	graph->array = (struct AdjList*)malloc(V * sizeof(struct AdjList));

	// Initialize each adjacency list as empty by making head as NULL 
	for (int i = 0; i < V; ++i)
		graph->array[i].head = NULL;

	return graph;
}

void addEdge(struct Graph* graph, int src, int dest, int weight)
{
	// Add an edge from src to dest.  A new node is added to the adjacency 
	// list of src.  The node is added at the beginning 
	struct AdjListNode* newNode = newAdjListNode(dest, weight);
	newNode->next = graph->array[src].head;
	graph->array[src].head = newNode;

	// Since graph is undirected, add an edge from dest to src also 
	newNode = newAdjListNode(src, weight);
	newNode->next = graph->array[dest].head;
	graph->array[dest].head = newNode;
}

struct MinHeapNode {
	int v;
	int key;
};

struct MinHeap {
	int size; // Number of heap nodes present currently 
	int capacity; // Capacity of min heap 
	int* pos; // This is needed for decreaseKey() 
	struct MinHeapNode** array;
	~MinHeap()
	{
		delete[] array;
	}
};

struct MinHeapNode* newMinHeapNode(int v, int key)
{
	struct MinHeapNode* minHeapNode = (struct MinHeapNode*)malloc(sizeof(struct MinHeapNode));
	minHeapNode->v = v;
	minHeapNode->key = key;
	return minHeapNode;
}

struct MinHeap* createMinHeap(int capacity)
{
	struct MinHeap* minHeap = (struct MinHeap*)malloc(sizeof(struct MinHeap));
	minHeap->pos = (int*)malloc(capacity * sizeof(int));
	minHeap->size = 0;
	minHeap->capacity = capacity;
	minHeap->array = (struct MinHeapNode**)malloc(capacity * sizeof(struct MinHeapNode*));
	return minHeap;
}

void swapMinHeapNode(struct MinHeapNode** a, struct MinHeapNode** b)
{
	struct MinHeapNode* t = *a;
	*a = *b;
	*b = t;
}

void minHeapify(struct MinHeap* minHeap, int idx)
{
	int smallest, left, right;
	smallest = idx;
	left = 2 * idx + 1;
	right = 2 * idx + 2;

	if (left < minHeap->size && minHeap->array[left]->key < minHeap->array[smallest]->key)
		smallest = left;

	if (right < minHeap->size && minHeap->array[right]->key < minHeap->array[smallest]->key)
		smallest = right;

	if (smallest != idx) {
		// The nodes to be swapped in min heap 
		MinHeapNode* smallestNode = minHeap->array[smallest];
		MinHeapNode* idxNode = minHeap->array[idx];

		// Swap positions 
		minHeap->pos[smallestNode->v] = idx;
		minHeap->pos[idxNode->v] = smallest;

		// Swap nodes 
		swapMinHeapNode(&minHeap->array[smallest], &minHeap->array[idx]);

		minHeapify(minHeap, smallest);
	}
}

int isEmpty(struct MinHeap* minHeap)
{
	return minHeap->size == 0;
}

struct MinHeapNode* extractMin(struct MinHeap* minHeap)
{
	if (isEmpty(minHeap))
		return NULL;

	// Store the root node 
	struct MinHeapNode* root = minHeap->array[0];

	// Replace root node with last node 
	struct MinHeapNode* lastNode = minHeap->array[minHeap->size - 1];
	minHeap->array[0] = lastNode;

	// Update position of last node 
	minHeap->pos[root->v] = minHeap->size - 1;
	minHeap->pos[lastNode->v] = 0;

	// Reduce heap size and heapify root 
	--minHeap->size;
	minHeapify(minHeap, 0);

	return root;
}

void decreaseKey(struct MinHeap* minHeap, int v, int key)
{
	// Get the index of v in  heap array 
	int i = minHeap->pos[v];

	// Get the node and update its key value 
	minHeap->array[i]->key = key;

	// Travel up while the complete tree is not hepified. 
	// This is a O(Logn) loop 
	while (i && minHeap->array[i]->key < minHeap->array[(i - 1) / 2]->key) {
		// Swap this node with its parent 
		minHeap->pos[minHeap->array[i]->v] = (i - 1) / 2;
		minHeap->pos[minHeap->array[(i - 1) / 2]->v] = i;
		swapMinHeapNode(&minHeap->array[i], &minHeap->array[(i - 1) / 2]);

		// move to parent index 
		i = (i - 1) / 2;
	}
}

bool isInMinHeap(struct MinHeap* minHeap, int v)
{
	if (minHeap->pos[v] < minHeap->size)
		return true;
	return false;
}


void printArr(int arr[], int n)
{
	for (int i = 1; i < n; ++i)
		printf("%d - %d\n", arr[i], i);
}

int* PrimMST(struct Graph* graph)
{
	int V = graph->V; // Get the number of vertices in graph 
	int* parent = new int[V]; // Array to store constructed MST 
	int* key = new int[V]; // Key values used to pick minimum weight edge in cut 

				// minHeap represents set E 
	struct MinHeap* minHeap = createMinHeap(V);

	// Initialize min heap with all vertices. Key value of 
	// all vertices (except 0th vertex) is initially infinite 
	int init = rand() % V;
	for (int v = 0; v < V; ++v) {
		if (v == init) {
			continue;
		}
		parent[v] = -1;
		key[v] = INT_MAX;
		if (v == 0) {
			minHeap->pos[v] = init;
			minHeap->array[init] = newMinHeapNode(v, key[v]);
		}
		else {
			minHeap->pos[v] = v;
			minHeap->array[v] = newMinHeapNode(v, key[v]);
		}
	}
	// Make key value of 0th vertex as 0 so that it 
	// is extracted first 
	key[init] = 0;
	minHeap->array[0] = newMinHeapNode(init, key[init]);
	minHeap->pos[init] = 0;

	// Initially size of min heap is equal to V 
	minHeap->size = V;
	minHeapify(minHeap, 0);

	// In the followin loop, min heap contains all nodes 
	// not yet added to MST. 
	while (!isEmpty(minHeap)) {
		// Extract the vertex with minimum key value 
		struct MinHeapNode* minHeapNode = extractMin(minHeap);
		int u = minHeapNode->v; // Store the extracted vertex number 
		delete minHeapNode;

								// Traverse through all adjacent vertices of u (the extracted 
								// vertex) and update their key values 
		struct AdjListNode* pCrawl = graph->array[u].head;
		while (pCrawl != NULL) {
			int v = pCrawl->dest;

			// If v is not yet included in MST and weight of u-v is 
			// less than key value of v, then update key value and 
			// parent of v 
			if (isInMinHeap(minHeap, v) && pCrawl->weight < key[v]) {
				key[v] = pCrawl->weight;
				parent[v] = u;
				decreaseKey(minHeap, v, key[v]);
			}
			pCrawl = pCrawl->next;
		}
	}
	delete[] key;
	delete minHeap;
	// print edges of MST 
	//printArr(parent, 10);
	return parent;
}
