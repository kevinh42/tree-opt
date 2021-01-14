#pragma once
#include <Eigen/Core>
#include <Eigen/SVD>
#include <vector>
#include <unordered_set>
#include <numeric>
#include <iterator>
#include <queue>
#include <iostream>

template <int _Dims>
class Optimize {
public:
    typedef Eigen::Array<float,_Dims,1> vars_type;
    typedef std::function<float(vars_type)> funk_type;

    Optimize(funk_type funk, vars_type min_bounds = -vars_type::Ones(), vars_type max_bounds = vars_type::Ones());
    ~Optimize() {};

    void setNumObservations(int num_observations) {num_observations_ = num_observations;};
    void setMaxDepth(int max_depth) {max_depth_ = max_depth;};
    void setThreshold(float threshold) {threshold_ = threshold;};
    void setMaxIterations(int max_iterations) {max_iterations_=max_iterations;};

    Optimize<_Dims>::vars_type run();

private:

    static int powint(int x, int p)//https://stackoverflow.com/a/1505791
    {
        if (p == 0) return 1;
        if (p == 1) return x;
        int tmp = powint(x, p/2);
        if (p%2 == 0) return tmp * tmp;
        else return x * tmp * tmp;
    }

    struct Split {
        int dimension;
        int left_index;
        int right_index;
        bool left_right; //Keep left or right side of split
        Split(int dim, int left_idx, int right_idx, bool lr) : dimension(dim), left_index(left_idx), right_index(right_idx), left_right(lr){};
    };

    struct Partition {
        std::vector<Split> splits;
        std::vector<int> elements;

        bool getClass(const Eigen::Matrix<float,-1,_Dims+1> & data, float threshold){
            int low = 0;
            for (auto ind : elements) {
                if (data(ind,data.cols()-1) < threshold)
                    low++;
            }
            return ((float) low / (float) elements.size())<=0.5;
        };

        void getBounds(const Eigen::Matrix<float,-1,_Dims+1> & data, vars_type & min, vars_type & max){
            for (auto split : splits){
                float x_left = data(split.left_index,split.dimension);
                float x_right = data(split.right_index,split.dimension);
                float x_mid = (x_left+x_right)/2;
                if (split.left_right) {
                    if (min(split.dimension) < x_mid)
                        min(split.dimension) = x_mid;
                } else {
                    if (max(split.dimension) > x_mid)
                        max(split.dimension) = x_mid;
                }
            }
        };
    };
    
    funk_type funk_;
    vars_type min_bounds_;
    vars_type max_bounds_;

    int num_observations_ = 125;
    float threshold_ = 3;
    int max_depth_ = 10;
    int max_iterations_ = 10;
    float tolerance_ = 1e-6;

    Eigen::Matrix<float,-1,_Dims+1> data_;
    Eigen::Matrix<float,_Dims,_Dims> householder_ = Eigen::Matrix<float,_Dims,_Dims>::Identity();
    
    //Sample randomly from bounds
    Eigen::Matrix<float,-1,_Dims+1> randSample(int n);
    Eigen::Matrix<float,-1,_Dims+1> randSampleLow(int n, const std::vector<vars_type> & mins, const std::vector<vars_type> & maxs);

    //Generates observations from random samples
    void generateData() {
        data_ = randSample(num_observations_); 
    };
    void generateDataLow(const std::vector<vars_type> & mins, const std::vector<vars_type> & maxs) {
        data_ = randSampleLow(num_observations_, mins, maxs); 
    };

    //Sorts along each dimension and stores permutation
    void sortData();
    std::array<std::vector<int>,_Dims> permutations_;


    float calculateGini(const std::vector<bool> & classified, int start, int end);
    int findSplit(const Partition & main_part, Partition & left_part, Partition & right_part);

    bool withinUnion(const vars_type & x, const std::vector<vars_type> & mins, const std::vector<vars_type> & maxs);

};

// Implementation
template <int _Dims>
Optimize<_Dims>::Optimize(Optimize::funk_type funk, Optimize::vars_type min_bounds, Optimize::vars_type max_bounds) :
    funk_(funk), 
    min_bounds_(min_bounds), 
    max_bounds_(max_bounds)
{
}

template <int _Dims>
typename Optimize<_Dims>::vars_type Optimize<_Dims>::run()
{
    //place box
    generateData();

    //partition box
    std::vector<int> v(num_observations_);
    std::iota(v.begin(),v.end(),0);

    vars_type current_min = vars_type::Zero();
    float current_min_f = std::numeric_limits<float>::max();

    for (int k = 0; k < max_iterations_; k++){
        //PCA to find dominant direction of hyper-ellipse
        Eigen::JacobiSVD<Eigen::Matrix<float,-1,_Dims>> svd(data_.block(0,0,num_observations_,_Dims), Eigen::ComputeThinU | Eigen::ComputeThinV);
        svd.computeV();
        vars_type dominant = svd.matrixV().template block<1,_Dims>(0,0).transpose();
        //Get Householder transformation matrix
        vars_type e1 = vars_type::Zero();
        e1(0) = 1;
        vars_type u = e1 - dominant;
        u.matrix().normalize();
        householder_ = Eigen::Matrix<float,_Dims,_Dims>::Identity() - 2*u.matrix() * u.matrix().transpose();
        
        //Apply Householder transformation
        bool no_transform = false;
        if (!std::isnan(householder_.norm()) && !std::isnan(data_.block(0,0,num_observations_,_Dims).norm())){
            data_.block(0,0,num_observations_,_Dims) = data_.block(0,0,num_observations_,_Dims).matrix() * householder_.transpose();
        } else {
            no_transform = true;
        }        
        sortData();
        
        std::queue<Partition> partition_queue;
        std::vector<Partition> terminal;

        Partition initial;
        initial.elements = v;
        partition_queue.push(initial);

        int depth = 0;
        while (depth < max_depth_){
            std::queue<Partition> temp_queue;
            while (!partition_queue.empty()){
                auto part = partition_queue.front();
                partition_queue.pop();

                //Check stopping conditions
                if (part.elements.size() <= 1){
                    //terminal
                    terminal.push_back(part);
                    continue;
                }

                Partition left_part, right_part;
                int split_ret = this->findSplit(part, left_part, right_part);

                if (!split_ret) {
                    terminal.push_back(part);
                } else if (depth == max_depth_-1){
                    terminal.push_back(left_part);
                    terminal.push_back(right_part);
                } else {
                    temp_queue.push(left_part);
                    temp_queue.push(right_part);
                }
            }
            partition_queue = temp_queue;
            depth++;
        }

        std::vector<vars_type> minimums;
        std::vector<vars_type> maximums;
        //Find bounding box in new coordinates
        //We have 2^_Dims corners
        vars_type new_min = householder_.transpose() * min_bounds_.matrix();
        vars_type new_max = householder_.transpose() * max_bounds_.matrix();
        for(int i = 0; i < powint(2,_Dims); i++) {
            vars_type corner;
            for (int j = 0; j < _Dims; j++){
                if (i & powint(2,j))
                    corner(j) = max_bounds_(j);
                else 
                    corner(j) = min_bounds_(j);
            }
            vars_type corner_transformed = householder_.transpose() * corner.matrix();
            for (int d = 0; d < _Dims; d++){
                if (corner_transformed(d) < new_min(d))
                    new_min(d) = corner_transformed(d);
                if (corner_transformed(d) > new_max(d))
                    new_max(d) = corner_transformed(d);
            }
        }
        for (auto t : terminal) {
            if (!t.getClass(data_,threshold_)){ //classified as low
                vars_type min = new_min;
                vars_type max = new_max;
                t.getBounds(data_, min, max);
                minimums.push_back(min);
                maximums.push_back(max);
            }
        }
 
        //Randomly sample in union of low partitions
        generateDataLow(minimums,maximums);
        
        //Apply inverse Householder transformation
        if (!no_transform){
            data_.block(0,0,num_observations_,_Dims) = data_.block(0,0,num_observations_,_Dims).matrix() * householder_;
        }

        for (int i = 0; i < num_observations_; i++){
            //Calculate f for new batch of points
            data_(i,_Dims) = funk_(data_.template block<1,_Dims>(i,0));
            //Check if f is new minimum
            if (data_(i,_Dims) < current_min_f){
                current_min = data_.template block<1,_Dims>(i,0);
                current_min_f = data_(i,_Dims);
            } 
        }
    }
    return current_min;
}

template <int _Dims>
Eigen::Matrix<float,-1,_Dims+1> Optimize<_Dims>::randSample(int n)
{
    Eigen::Matrix<float,-1,_Dims+1> samples;
    samples.resize(n,_Dims+1);
    for (int i = 0; i < n; i++){
        vars_type x = vars_type::Random();//between -1 and 1
        x += 1;
        x /= 2;
        x *= max_bounds_-min_bounds_;
        x += min_bounds_;
        samples.template block<1,_Dims>(i,0) = x;
        samples(i,_Dims) = funk_(x);
    }
    return samples;
}

template <int _Dims>
Eigen::Matrix<float,-1,_Dims+1> Optimize<_Dims>::randSampleLow(int n, const std::vector<vars_type> & mins, const std::vector<vars_type> & maxs)
{
    Eigen::Matrix<float,-1,_Dims+1> samples;
    samples.resize(n,_Dims+1);

    //Create a new bounding box on which to sample
    // vars_type new_min_bounds = mins[0];
    // vars_type new_max_bounds = maxs[0];
    vars_type new_min_bounds = mins[0];
    vars_type new_max_bounds = maxs[0];

    for (int i = 1; i < mins.size(); i++){
        for (int d = 0; d < _Dims; d++){
            if (mins[i](d) < new_min_bounds(d))
                new_min_bounds(d) = mins[i](d);
            if (maxs[i](d) > new_max_bounds(d))
                new_max_bounds(d) = maxs[i](d);
        }
    }

    for (int i = 0; i < n; i++){
        vars_type x = vars_type::Random();//between -1 and 1
        x += 1;
        x /= 2;
        x *= new_max_bounds-new_min_bounds;
        x += new_min_bounds;
        while (!withinUnion(x, mins, maxs)){
            x = vars_type::Random();//between -1 and 1
            x += 1;
            x /= 2;
            x *= new_max_bounds-new_min_bounds;
            x += new_min_bounds;
        }
        samples.template block<1,_Dims>(i,0)= x;
    }
    return samples;
}

template <int _Dims>
float Optimize<_Dims>::calculateGini(const std::vector<bool> & classified, int start, int end)
{
    int num_low_int=0;
    for (int i = start; i < end; i++){
        if (!classified[i])
            num_low_int++;
    }
    float num_low = num_low_int;
    float num_total = end-start;

    return 2*num_low/num_total * (1 - num_low/num_total);
}

template <int _Dims>
int Optimize<_Dims>::findSplit(const Optimize<_Dims>::Partition & main_part, Optimize<_Dims>::Partition & left_part, Optimize<_Dims>::Partition & right_part)
{
    auto indices = main_part.elements;

    std::vector<int> v(data_.rows());
    std::iota(v.begin(),v.end(),0);

    //classify indices
    std::unordered_set<int> low;
    std::unordered_set<int> high;
    
    for (int i : indices){
        if (data_(i,_Dims) < threshold_)
            low.insert(i);
        else
            high.insert(i);    
    }

    //partition
    float delta = 0;
    for (int dim = 0; dim < _Dims; dim++){
        auto p = permutations_[dim];
        //keep only indices in current partition
        for (auto it = p.begin(); it != p.end(); ){
            if (low.find(*it)==low.end() && high.find(*it)==high.end())
                it = p.erase(it);
            else
                ++it;
        }
        //sort indices in non-decreasing order along current dimension
        std::vector<int> sorted_inds(indices.size());
        std::transform(p.begin(), p.end(), sorted_inds.begin(), [&](int i){ return v[i]; });
         //get class at each index
        std::vector<bool> classified;
        for (int ind : sorted_inds){
            if (low.find(ind) == low.end())
                classified.push_back(true); //high
            else
                classified.push_back(false); //low
        }
        //find best split
        for (int i = 0; i < classified.size() - 1; i++){
            float gini_left = calculateGini(classified, 0, i+1);
            float gini_right = calculateGini(classified, i+1, classified.size());
            float gini_all = calculateGini(classified, 0, classified.size());

            float size_left = i+1;
            float size_right = classified.size() - i - 1;
            float size_all = classified.size();

            float d = gini_all - gini_left*size_left/size_all - gini_right*size_right/size_all;

            if (d > delta){
                delta = d;
                left_part = main_part;
                right_part = main_part;

                left_part.elements =std::vector<int>(sorted_inds.begin(),std::next(sorted_inds.begin(),i+1));
                right_part.elements = std::vector<int>(std::next(sorted_inds.begin(),i+1),sorted_inds.end());

                left_part.splits.push_back(Split(dim,sorted_inds[i],sorted_inds[i+1],false));
                right_part.splits.push_back(Split(dim,sorted_inds[i],sorted_inds[i+1],true));
            }
        }
    }
    //Return 0 if stopping condition has been met
    if (delta < tolerance_){
        return 0;
    }
    return 1;
}

template <int _Dims>
void Optimize<_Dims>::sortData()
{
    //Each vector in permutations_ should contain the sorted indices along a dimension
    for (int dim = 0; dim < _Dims; dim++){
        std::vector<float> xi(data_.rows());
        for (int i = 0; i < data_.rows(); i++)
            xi[i] = data_(i,dim);
        //sort in non-decreasing order
        std::vector<int> p(xi.size());
        std::iota(p.begin(), p.end(), 0);
        std::sort(p.begin(), p.end(), [&](std::size_t i, std::size_t j){ return xi[i] < xi[j]; });
        //store permutation
        permutations_[dim] = p;
    }
}

template <int _Dims>
bool Optimize<_Dims>::withinUnion(const vars_type & x, const std::vector<vars_type> & mins, const std::vector<vars_type> & maxs)
{
    for (int i = 0; i < mins.size(); i++){
        if ((x-mins[i] < 0).count() == 0 && (maxs[i]-x < 0).count() == 0){
            return true;
        }
    }
    return false;
}