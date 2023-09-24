#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <ctime>
#include <float.h>
#include <math.h>

//make function for reading data from whatever format later

std::mt19937_64 generator(static_cast<unsigned long>(std::time(0)));
std::uniform_real_distribution<double> distribution(0.0, 10.0);
// quick fisher-yates courtesy of chatgpt
std::vector<int> fisherYatesShuffle(int n, int k) {
    std::vector<int> indices(n); // Array to hold indices [0, 1, ..., n-1]

    // Initialize the array with indices
    for (int i = 0; i < n; ++i) {
        indices[i] = i;
    }

    // Fisher-Yates Shuffle
    for (int i = 0; i < k; ++i) {
        std::uniform_int_distribution<int> distribution(i, n - 1); // Create a uniform distribution
        int j = distribution(generator); // Generate a random index

        // Swap indices[i] and indices[j]
        std::swap(indices[i], indices[j]);
    }

    // Resize the vector to keep only the first k shuffled indices
    indices.resize(k);

    return indices;
}

double euc_dist(std::vector<double>* vec1, std::vector<double>* vec2)
{
    double dist = 0.0;
    for (int i = 0; i < (*vec1).size(); i++)
    {
        dist += pow(((*vec1)[i]-(*vec2)[i]), 2);
    }
    return sqrt(dist);
}

std::vector<double>* randvec(int dim)
{
    std::vector<double>* thevec = new std::vector<double>(dim);
    for (int i = 0; i < dim; i++)
    {
        (*thevec)[i] = distribution(generator);
    }
    return thevec;
}

std::vector<double>* zerovec(int dim)
{
    std::vector<double>* thevec = new std::vector<double>(dim);
    for (int i = 0; i < dim; i++)
    {
        (*thevec)[i] = 0.0;
    }
    return thevec;
}

void dataprint(std::vector<std::vector<double>*> &thevec)
{
    for (int i = 0; i < thevec.size(); i++)
    {
        for (int j = 0; j < (*thevec[0]).size(); j++)
        {
            std::cout << i << ":  " << (*thevec[i])[j] << "  ";
        }
        std::cout << "\n";
    }
}

// thank you Gpt4 <3
// Function to read a CSV file and return a vector of pointers to vectors of doubles
std::vector<std::vector<double>*> readCSV(const std::string &filePath) {
    std::vector<std::vector<double>*> data;
    std::ifstream file(filePath);
    std::string line;
    std::string token;

    // Check if the file is open
    if (!file.is_open()) {
        std::cerr << "Could not open the file: " << filePath << std::endl;
        return data;
    }

    // Read each line from the file
    while (std::getline(file, line)) {
        std::vector<double> *row = new std::vector<double>;
        std::istringstream iss(line);
        
        // Read each token separated by a comma
        while (std::getline(iss, token, ',')) {
            row->push_back(std::stod(token));
        }

        data.push_back(row);
    }

    return data;
}

void update(std::vector<std::vector<double>*> &centroids, std::vector<std::vector<double>*> &data)
{
    //we make a thing of initial centroids (vector of vec pointers)
    //we want to update these:
    //we figure out the closest centroid for each of the datapoints, store idx of each of those in an array of length (samples)
    //iterate through that, count up num of samples for each point, add those samples together, divide by that number (we can start overwriting)
    std::vector<int> closest(data.size());
    std::vector<int> count(centroids.size(), 0);
    for (int i = 0; i < data.size(); i++)
    {
        double themin = DBL_MAX;
        for (int j = 0; j < centroids.size(); j++)
        {
            double dist = euc_dist(data[i], centroids[j]);
            if (dist < themin)
            {
                themin = dist;
                closest[i] = j;
            }
        }
        count[closest[i]]++;
    }
    //ok we have vector of counts, closest centroid for each datapoint
    //we can start calculating new centroids by clearing out prev centroids,
    int dim = (*centroids[0]).size();
    
    for (int i = 0; i < centroids.size(); i++)
    {
        delete centroids[i];
        centroids[i] = zerovec(dim);
    }
    
    for (int i = 0; i < closest.size(); i++)
    {
        //centroid closest to data[i] (centroid at k)
        //add data[i] to that centroid (we normalize aferwards with counts)
        for (int j = 0; j < dim; j++)
        {
            //std::cout << (*data[i])[j] << "\n";
            //std::cout << (*centroids[closest[i]])[j] << "\n";
            (*centroids[closest[i]])[j] += (*data[i])[j];
            //std::cout << (*centroids[closest[i]])[j] << "\n";
        }
    }
    //normalize by counts for each centroid
    for (int i = 0; i < centroids.size(); i++)
    {
        for (int j = 0; j < dim; j++)
        {
            (*centroids[i])[j] /= double(count[i]);
        }
    }
}

int main()
{
    /*
    ok so we have a bunch of n-d vectors, and we have k. 
    we first need to assign the centroids randomly.
    we will either use the k means ++ implentation or the forgy method. maybe make it so that we can switch between.
    start with forgy method
    we
    */
    // figure out how to load the data into a standard vector of standard vectors for now, optimize later
    int num_clusters = 3;
    /*
    //OG
    int dim = 3;
    int samples = 100;
    int num_clusters = 8;
    

    std::vector<std::vector<double>*> thedata(samples);
    //lets have randomly generated vectors, something chill like 3d for now. let's make 100 of these
    for (int i = 0; i < 100; i++)
    {   //std::cout << i;
        thedata[i] = randvec(dim);
    }
    */
    //dataprint(thedata);
    //we have our samples now, create function/stuff for updating

    //we make a thing of initial centroids (vector of vec pointers)
    //we want to update these:
    //we figure out the closest centroid for each of the datapoints, store each of those in an array of length (samples)
    //iterate through that, count up num of samples for each point, add those samples together, divide by that number (we can start overwriting)

    //so the function takes in the vec of centroids, vec of points, outputs vec of centroids

    //gettting the first random initializations

    std::vector<std::vector<double>*> thedata;
    thedata = readCSV("iris_edit.csv");
    //dataprint(thedata);

    std::vector<int> indices = fisherYatesShuffle(thedata.size(), num_clusters);
    std::vector<std::vector<double>*> centroids(num_clusters);
    for (int i = 0; i < num_clusters; i++)
    {
        centroids[i] = new std::vector<double>(*thedata[indices[i]]);
    }
    //dataprint(centroids);

    int num_iterations = 100;
    for (int i = 0; i < num_iterations; i++)
    {
        update(centroids, thedata);
        std::cout << "iteration " << i << "\n";
        dataprint(centroids);
        std::cout << "///////////////////////////////////////////////" << "\n";
    }
}
